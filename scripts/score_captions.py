"""Real-metric scoring of a saved adapter on a caption corpus val split.

The paper-plan §C3 metric run: loads a saved bridged-GiT + LoRA adapter,
generates captions for the FULL val split under each configured decode
variant, and scores them with multi-reference BLEU-1..4, CIDEr,
BERTScore-bn, and (optionally) M-CLIPScore.

Corpus-agnostic via ``data.loader`` (banglalekha | bancap) — BAN-Cap
items carry all 5 annotator references; BanglaLekha items fall back to
the single selected caption as reference.

VRAM discipline (8 GB local GPU): the captioner is loaded for the
generation phase only and freed before the metric models (mBERT for
BERTScore, CLIP towers for M-CLIPScore) are loaded.

Relative data/adapter paths in the config are resolved against the repo
root, so the same config works on any machine when run from anywhere.

Run:
    python scripts/score_captions.py --config configs/score_captions_bancap.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from src.data import build_caption_dataset  # noqa: E402
from src.evaluation.caption_metrics import (  # noqa: E402
    bertscore_bn,
    m_clipscore,
    ngram_scores,
)
from src.models.adapter_io import load_bridged_lora  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


@torch.no_grad()
def generate_all(model, processor, dataset, params: dict, batch_size: int,
                 max_new_tokens: int, device) -> list[str]:
    """Generate one caption per dataset item, batched."""
    hyps: list[str] = []
    n = len(dataset)
    t0 = time.time()
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        pixel_values = processor(
            images=[it.image for it in items], return_tensors="pt"
        ).pixel_values.to(device)
        gen_ids = model.generate(
            pixel_values=pixel_values, max_new_tokens=max_new_tokens, **params
        )
        hyps.extend(processor.batch_decode(gen_ids, skip_special_tokens=True))
        done = min(start + batch_size, n)
        if start // batch_size % 5 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    {done}/{n} ({rate:.1f} img/s)")
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[cfg["model"]["dtype"]]

    adapter_dir = paths.EXP_CHECKPOINTS / cfg["adapter"]["checkpoint_subdir"]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")
    print(f"[adapter] {adapter_dir}")

    dcfg = cfg["data"]
    images_dir = resolve_path(dcfg["images_dir"])
    val_ds = build_caption_dataset(
        dcfg["loader"],
        captions_path=resolve_path(dcfg["captions_path"]),
        images_dir=images_dir,
        split="val",
        val_fraction=dcfg["val_fraction"],
        max_samples=dcfg.get("max_val_samples"),
        caption_selection=dcfg.get("caption_selection", "first"),
        seed=cfg["seed"],
    )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"val split is empty — check images_dir={images_dir} "
            "(the loader silently drops rows whose image file is missing)"
        )
    print(f"[data] loader={dcfg['loader']}  val={len(val_ds)}")

    # References + image paths, gathered once (order == dataset order).
    refs: list[list[str]] = []
    filenames: list[str] = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        refs.append(list(getattr(item, "references", None) or [item.caption]))
        filenames.append(item.filename)
    image_paths = [images_dir / f for f in filenames]
    n_refs = sorted({len(r) for r in refs})
    print(f"[data] references per image: {n_refs}")

    # ---- Phase 1: generation (captioner on GPU) ----------------------
    print(f"[model] loading {cfg['model']['name']} + bridge + adapter ...")
    model, processor = load_bridged_lora(
        cfg["model"]["name"], adapter_dir, dtype, device
    )
    print(f"  vocab_size={len(processor.tokenizer)}")

    gcfg = cfg["generation"]
    predictions: dict[str, list[str]] = {}
    for variant in cfg["variants"]:
        set_seed(cfg["seed"])  # sampling variants stay reproducible
        name = variant["name"]
        params = dict(variant["params"])
        print(f"[generate] variant={name}  params={params}")
        predictions[name] = generate_all(
            model, processor, val_ds, params,
            batch_size=gcfg["batch_size"],
            max_new_tokens=gcfg["max_new_tokens"],
            device=device,
        )

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Persist predictions BEFORE scoring so a metric failure never
    # discards the (expensive) generation phase.
    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps(
            {
                "filenames": filenames,
                "references": refs,
                "predictions": predictions,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # ---- Phase 2: metrics (metric models on GPU, one at a time) ------
    mcfg = cfg["metrics"]
    metrics: dict[str, dict[str, float]] = {}
    for name, hyps in predictions.items():
        print(f"[score] variant={name}")
        row: dict[str, float] = {}
        row["mean_len_words"] = sum(len(h.split()) for h in hyps) / len(hyps)
        n_empty = sum(1 for h in hyps if not h.strip())
        row["n_empty"] = float(n_empty)
        if n_empty:
            print(f"  [warn] {n_empty} empty generations in this variant")

        ng = ngram_scores(hyps, refs)
        row.update(bleu1=ng.bleu1, bleu2=ng.bleu2, bleu3=ng.bleu3,
                   bleu4=ng.bleu4, cider=ng.cider)

        if mcfg.get("bertscore", True):
            # bert-score crashes on empty hypotheses (and empty strings
            # are also undefined for it semantically) — substitute a bare
            # danda; n_empty above keeps the report honest.
            hyps_bs = [h if h.strip() else "।" for h in hyps]
            row.update(bertscore_bn(
                hyps_bs, refs, device=str(device),
                batch_size=mcfg.get("bertscore_batch_size", 64),
            ))
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if mcfg.get("clipscore", True):
            try:
                row["m_clipscore"] = m_clipscore(
                    image_paths, hyps,
                    text_model_name=mcfg["clip_text_model"],
                    image_model_name=mcfg["clip_image_model"],
                    device=str(device),
                    batch_size=mcfg.get("clipscore_batch_size", 32),
                )
            except ImportError:
                print("  [warn] sentence-transformers not installed — "
                      "skipping M-CLIPScore")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metrics[name] = row
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))

    # ---- Outputs (predictions.json already written post-generation) ---
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_cols = list(next(iter(metrics.values())).keys())
    md = [
        f"# Caption metrics — `{cfg['adapter']['checkpoint_subdir']}` adapter",
        "",
        f"Corpus: `{dcfg['loader']}` val split, {len(val_ds)} images, "
        f"{n_refs} references/image (seed={cfg['seed']}, "
        f"val_fraction={dcfg['val_fraction']}).",
        f"Scorers: pycocoevalcap BLEU/CIDEr (whitespace+danda tokenization), "
        f"BERTScore lang=bn, M-CLIPScore (w=2.5, multilingual text tower).",
        "",
        "| Variant | " + " | ".join(metric_cols) + " |",
        "|---|" + "|".join("---:" for _ in metric_cols) + "|",
    ]
    for name, row in metrics.items():
        md.append(
            f"| `{name}` | "
            + " | ".join(f"{row.get(c, float('nan')):.4f}" for c in metric_cols)
            + " |"
        )
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nWrote: {out_dir / 'results.md'}")
    print(f"       {out_dir / 'metrics.json'}")
    print(f"       {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
