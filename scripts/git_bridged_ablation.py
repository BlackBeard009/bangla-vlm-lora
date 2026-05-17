"""Bridged-GiT + LoRA: vocabulary extension × embedding-init ablation.

For each init strategy in the config, load a fresh GiT-base, bridge its
vocabulary with Bangla tokens from a donor tokenizer, apply LoRA (text-side
attention + modules_to_save on LM head and word embeddings), train, and
generate.

Run from repo root:
    python scripts/git_bridged_ablation.py --config configs/git_bridged_ablation.yaml
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
from src.tokenizer.bridge import bridge_vocabulary  # noqa: E402


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
    init_strategy: str
    vocab_size_before: int = 0
    vocab_size_after: int = 0
    n_added: int = 0
    trainable: int = 0
    trainable_pct: float = 0.0
    loss_start: float = float("nan")
    loss_end: float = float("nan")
    wall_seconds: float = 0.0
    generated: str = ""
    losses: list[float] = field(default_factory=list)


def run_variant(init_strategy: str, cfg: dict, dataset, device) -> VariantResult:
    print(f"\n=== bridged-GiT  init={init_strategy} ===")
    set_seed(cfg["seed"])

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    bundle = load_git(
        cfg["model"]["name"],
        dtype=dtype_map[cfg["model"]["dtype"]],
        use_fast_tokenizer=cfg["model"].get("use_fast_tokenizer", True),
    )

    # 1. Bridge vocabulary
    report = bridge_vocabulary(
        bundle,
        donor_tokenizer=cfg["bridge"]["donor_tokenizer"],
        donor_model_for_init=cfg["bridge"]["donor_model_for_init"],
        init_strategy=init_strategy,
    )
    print(
        f"  vocab: {report.n_existing} -> {report.n_final} "
        f"(+{report.n_added} Bangla tokens from {report.donor})"
    )

    # 2. Apply LoRA
    bundle = apply_lora(
        bundle,
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        modules_to_save=cfg["lora"]["modules_to_save"],
    )
    bundle.model.to(device)
    bundle.model.train()

    p = trainable_param_summary(bundle.model)
    print(f"  trainable: {p['trainable']:,} / {p['total']:,}  ({p['trainable_pct']:.3f}%)")

    result = VariantResult(
        init_strategy=init_strategy,
        vocab_size_before=report.n_existing,
        vocab_size_after=report.n_final,
        n_added=report.n_added,
        trainable=p["trainable"],
        trainable_pct=p["trainable_pct"],
    )

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

    bundle.model.eval()
    sample = dataset[0]
    pixel_values = bundle.processor(images=sample.image, return_tensors="pt").pixel_values.to(device)
    gen_ids = bundle.model.generate(
        pixel_values=pixel_values,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        num_beams=cfg["generation"]["num_beams"],
    )
    result.generated = bundle.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    del bundle, optim, trainable, loader
    torch.cuda.empty_cache()
    return result


def write_report(results: list[VariantResult], reference: str, out_path: Path) -> None:
    lines = [
        "# Bridged-GiT + LoRA ablation",
        "",
        f"Reference caption: `{reference}`",
        "",
        "| Init | Vocab before → after | Added | Trainable | % | Loss start→end | Wall (s) | Generated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.init_strategy}` | {r.vocab_size_before} → {r.vocab_size_after} | "
            f"+{r.n_added} | {r.trainable:,} | {r.trainable_pct:.2f}% | "
            f"{r.loss_start:.3f} → {r.loss_end:.3f} | {r.wall_seconds:.1f} | "
            f"`{r.generated}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bridge-ablation] device={device}, init strategies={cfg['init_strategies']}")
    dataset = SyntheticBanglaCaptions()

    results = [run_variant(s, cfg, dataset, device) for s in cfg["init_strategies"]]
    reference = dataset[0].caption

    print("\n" + "=" * 78)
    print("BRIDGED-GIT ABLATION SUMMARY")
    print("=" * 78)
    print(f"reference: {reference!r}")
    for r in results:
        print(
            f"  {r.init_strategy:8s}  vocab+{r.n_added:>5}  "
            f"trainable={r.trainable_pct:6.2f}%  "
            f"loss {r.loss_start:6.3f} -> {r.loss_end:6.3f}  "
            f"-> {r.generated!r}"
        )

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / "git_bridged_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(results, reference, out_dir / "results.md")
    print(f"\nWrote: {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
