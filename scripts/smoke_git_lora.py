"""Smoke test: GiT-base + LoRA on synthetic Bangla.

Mirrors scripts/smoke_git_train.py but wraps the model in a PEFT LoRA
adapter before training. Same dataset, same step count, same logging —
the only deltas are the LoRA wiring and a higher learning rate (only
adapter params train).

Success criteria:
  - LoRA wrapping completes
  - Trainable-param fraction is < 5% of base
  - Loss decreases (catches HF #1958 'doesn't learn' failure mode)
  - generate() returns non-empty output
  - Adapter saves and is small on disk
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
from src.models.git_lora import apply_lora, trainable_param_summary  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch, processor):
    images = [item.image for item in batch]
    captions = [item.caption for item in batch]
    enc = processor(images=images, text=captions, padding="longest", return_tensors="pt")
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
    print(f"[smoke-lora] device={device}")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    print(f"[smoke-lora] loading {cfg['model']['name']} ...")
    bundle = load_git(cfg["model"]["name"], dtype=dtype_map[cfg["model"]["dtype"]])

    print(f"[smoke-lora] applying LoRA: {cfg['lora']}")
    bundle = apply_lora(
        bundle,
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    bundle.model.to(device)
    bundle.model.train()

    p = trainable_param_summary(bundle.model)
    print(
        f"[smoke-lora] params: trainable={p['trainable']:,} / "
        f"total={p['total']:,}  ({p['trainable_pct']:.3f}%)"
    )
    # The HF #1958 concern would show up as either zero trainable params
    # or trainable params being on the wrong modules. Catch the former.
    assert p["trainable"] > 0, "LoRA wrapped but no parameters are trainable"
    assert p["trainable_pct"] < 5.0, "More than 5% trainable — adapter wired wrong?"

    dataset = SyntheticBanglaCaptions()
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, bundle.processor),
    )

    # Optimize only trainable params (PEFT freezes the base by default,
    # but be explicit so the optimizer's state matches what we report).
    trainable = [p for p in bundle.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    print(
        f"[smoke-lora] training {cfg['training']['steps']} steps on "
        f"{len(dataset)} samples, batch={cfg['training']['batch_size']}"
    )
    losses: list[float] = []
    step = 0
    t0 = time.time()
    while step < cfg["training"]["steps"]:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = bundle.model(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
            if step % cfg["training"]["log_every"] == 0:
                print(f"  step {step:>3}  loss={out.loss.item():.4f}")
            step += 1
            if step >= cfg["training"]["steps"]:
                break
    dt = time.time() - t0

    assert all(l == l for l in losses), "NaN loss"  # noqa: E741
    # The most important assertion: LoRA actually learned (HF #1958 fear).
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}. "
        "Reproduces HF peft #1958 'doesn't learn' concern."
    )

    # Generation sanity check
    bundle.model.eval()
    sample = dataset[0]
    pixel_values = bundle.processor(images=sample.image, return_tensors="pt").pixel_values.to(device)
    gen_ids = bundle.model.generate(
        pixel_values=pixel_values,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        num_beams=cfg["generation"]["num_beams"],
    )
    generated = bundle.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Save adapter (PEFT-style: only the LoRA weights)
    paths.ensure_dirs()
    ckpt_dir = paths.EXP_CHECKPOINTS / "smoke_git_lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(ckpt_dir)
    bundle.processor.save_pretrained(ckpt_dir)
    adapter_bytes = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())

    print()
    print("[smoke-lora] PASS")
    print(f"  steps          : {len(losses)}")
    print(f"  loss start->end: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  trainable      : {p['trainable']:,} ({p['trainable_pct']:.3f}%)")
    print(f"  wall time      : {dt:.1f}s")
    print(f"  reference cap  : {sample.caption!r}")
    print(f"  generated      : {generated!r}")
    print(f"  saved to       : {ckpt_dir}")
    print(f"  adapter size   : {adapter_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
