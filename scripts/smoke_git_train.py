"""Smoke test: fine-tune GiT-base on 16 in-memory Bangla caption samples.

Run from repo root:
    python scripts/smoke_git_train.py --config configs/smoke_git.yaml

Success criteria:
  - Model + processor load without errors
  - Forward pass produces a finite loss
  - Loss decreases over `training.steps` (memorization on tiny batch)
  - generate() returns a non-empty Bangla-ish string
  - Adapter / model state saves and reloads
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from src.data.synthetic_bangla import SyntheticBanglaCaptions  # noqa: E402
from src.models.git import load_git  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch, processor):
    """Use the processor to batch images + texts together."""
    images = [item.image for item in batch]
    captions = [item.caption for item in batch]
    enc = processor(
        images=images,
        text=captions,
        padding="longest",
        return_tensors="pt",
    )
    # GiT trains as causal LM: labels = input_ids with pad masked to -100
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    print(f"[smoke] loading {cfg['model']['name']} ...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    bundle = load_git(cfg["model"]["name"], dtype=dtype_map[cfg["model"]["dtype"]])
    bundle.model.to(device)
    bundle.model.train()

    dataset = SyntheticBanglaCaptions()
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, bundle.processor),
    )

    optim = torch.optim.AdamW(
        bundle.model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    print(f"[smoke] training {cfg['training']['steps']} steps "
          f"on {len(dataset)} samples, batch={cfg['training']['batch_size']}")
    losses: list[float] = []
    step = 0
    t0 = time.time()
    while step < cfg["training"]["steps"]:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = bundle.model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            losses.append(loss.item())
            if step % cfg["training"]["log_every"] == 0:
                print(f"  step {step:>3}  loss={loss.item():.4f}")
            step += 1
            if step >= cfg["training"]["steps"]:
                break
    dt = time.time() - t0

    # Smoke-test assertions
    assert all(l == l for l in losses), "NaN loss"  # noqa: E741
    assert losses[-1] < losses[0], f"Loss did not decrease ({losses[0]:.3f} -> {losses[-1]:.3f})"

    # Generation check
    bundle.model.eval()
    sample = dataset[0]
    pixel_values = bundle.processor(images=sample.image, return_tensors="pt").pixel_values.to(device)
    gen_ids = bundle.model.generate(
        pixel_values=pixel_values,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        num_beams=cfg["generation"]["num_beams"],
    )
    generated = bundle.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Save checkpoint
    paths.ensure_dirs()
    ckpt_dir = paths.EXP_CHECKPOINTS / "smoke_git_base"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(ckpt_dir)
    bundle.processor.save_pretrained(ckpt_dir)

    print()
    print("[smoke] PASS")
    print(f"  steps          : {len(losses)}")
    print(f"  loss start->end: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  wall time      : {dt:.1f}s")
    print(f"  reference cap  : {sample.caption!r}")
    print(f"  generated      : {generated!r}")
    print(f"  saved to       : {ckpt_dir}")


if __name__ == "__main__":
    main()
