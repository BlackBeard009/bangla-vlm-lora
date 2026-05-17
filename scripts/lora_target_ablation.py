"""LoRA target-module ablation on GiT-base + synthetic Bangla.

Runs every variant in the config from the same starting weights and
prints a side-by-side comparison table at the end.

Run from repo root:
    python scripts/lora_target_ablation.py --config configs/lora_target_ablation.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
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


@dataclass
class VariantResult:
    id: str
    label: str
    trainable: int = 0
    trainable_pct: float = 0.0
    loss_start: float = float("nan")
    loss_end: float = float("nan")
    wall_seconds: float = 0.0
    generated: str = ""
    losses: list[float] = field(default_factory=list)


def run_variant(variant: dict, cfg: dict, dataset, device) -> VariantResult:
    """Train one variant from a freshly-loaded model. Return per-variant metrics."""
    print(f"\n=== variant {variant['id']}: {variant['label']} ===")
    set_seed(cfg["seed"])  # deterministic across variants

    # Fresh model each variant — variants must not share weights.
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    bundle = load_git(cfg["model"]["name"], dtype=dtype_map[cfg["model"]["dtype"]])
    bundle = apply_lora(
        bundle,
        rank=variant["lora"]["rank"],
        alpha=variant["lora"]["alpha"],
        dropout=variant["lora"]["dropout"],
        target_modules=variant["lora"]["target_modules"],
        modules_to_save=variant["lora"]["modules_to_save"],
    )
    bundle.model.to(device)
    bundle.model.train()

    p = trainable_param_summary(bundle.model)
    print(f"  trainable: {p['trainable']:,} / {p['total']:,}  ({p['trainable_pct']:.3f}%)")

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, bundle.processor),
    )
    trainable = [pp for pp in bundle.model.parameters() if pp.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    result = VariantResult(
        id=variant["id"],
        label=variant["label"],
        trainable=p["trainable"],
        trainable_pct=p["trainable_pct"],
    )
    step = 0
    t0 = time.time()
    while step < cfg["training"]["steps"]:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = bundle.model(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            result.losses.append(out.loss.item())
            if step % cfg["training"]["log_every"] == 0:
                print(f"  step {step:>3}  loss={out.loss.item():.4f}")
            step += 1
            if step >= cfg["training"]["steps"]:
                break
    result.wall_seconds = time.time() - t0
    result.loss_start = result.losses[0]
    result.loss_end = result.losses[-1]

    # Generate from a held image
    bundle.model.eval()
    sample = dataset[0]
    pixel_values = bundle.processor(images=sample.image, return_tensors="pt").pixel_values.to(device)
    gen_ids = bundle.model.generate(
        pixel_values=pixel_values,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        num_beams=cfg["generation"]["num_beams"],
    )
    result.generated = bundle.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Free memory before next variant.
    del bundle, optim, trainable, loader
    torch.cuda.empty_cache()

    return result


def write_report(results: list[VariantResult], reference: str, out_path: Path) -> None:
    lines = [
        "# LoRA target-module ablation",
        "",
        f"Reference caption (held image): `{reference}`",
        "",
        "| Variant | Targets | Trainable | % of base | Loss start → end | Wall (s) | Generated |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.id}` | {r.label} | {r.trainable:,} | {r.trainable_pct:.3f}% | "
            f"{r.loss_start:.3f} → {r.loss_end:.3f} | {r.wall_seconds:.1f} | `{r.generated}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] device={device}, variants={len(cfg['variants'])}")
    dataset = SyntheticBanglaCaptions()

    results = [run_variant(v, cfg, dataset, device) for v in cfg["variants"]]
    reference = dataset[0].caption

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"reference: {reference!r}")
    for r in results:
        print(
            f"  {r.id:25s} trainable={r.trainable_pct:6.3f}%  "
            f"loss {r.loss_start:6.3f} -> {r.loss_end:6.3f}  "
            f"-> {r.generated!r}"
        )

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / "lora_target_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(results, reference, out_dir / "results.md")
    print(f"\nWrote: {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
