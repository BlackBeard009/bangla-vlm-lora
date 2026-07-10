"""No-vision trivial baselines for the metric-validity study.

Builds caption "systems" that never look at the image, generates their
outputs for a corpus val split, and scores them with the exact same
multi-reference battery as the real models (BLEU-1..4 / CIDEr /
BERTScore-bn / M-CLIPScore). If a no-vision baseline lands inside the
score range published for real models on the same corpus family, those
published scores measured the corpus prior, not visual captioning.

Baselines (all derived from the TRAIN split only):
  constant   : the single most frequent training caption, emitted for
               every test image.
  mode_len   : the most frequent training caption of the median
               reference length (a "length-matched template").
  random     : a uniformly random training caption per test image
               (seeded; no image information).
  freq_words : sentence assembled from the k most frequent training
               tokens in frequency order — the "pitfall words" string.

Corpus-agnostic via data.loader. CPU except BERTScore/M-CLIPScore.

Run:
    python scripts/trivial_baselines.py --config configs/trivial_baselines_bancap.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import Counter
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


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def build_baselines(train_captions: list[str], n_val: int, seed: int,
                    freq_k: int = 8) -> dict[str, list[str]]:
    counts = Counter(c.strip() for c in train_captions if c.strip())
    constant = counts.most_common(1)[0][0]

    lengths = sorted(len(c.split()) for c in counts)
    median_len = lengths[len(lengths) // 2]
    mode_len = next(
        (c for c, _ in counts.most_common()
         if len(c.split()) == median_len),
        constant,
    )

    rng = random.Random(seed)
    pool = list(counts)
    rand = [rng.choice(pool) for _ in range(n_val)]

    tok_counts = Counter(w for c in counts for w in c.split())
    freq_words = " ".join(w for w, _ in tok_counts.most_common(freq_k))

    return {
        "constant": [constant] * n_val,
        "mode_len": [mode_len] * n_val,
        "random": rand,
        "freq_words": [freq_words] * n_val,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]
    images_dir = resolve_path(dcfg["images_dir"])
    common = dict(
        captions_path=resolve_path(dcfg["captions_path"]),
        images_dir=images_dir,
        val_fraction=dcfg["val_fraction"],
        seed=cfg["seed"],
    )
    train_ds = build_caption_dataset(
        dcfg["loader"], split="train",
        caption_selection=dcfg.get("caption_selection", "random"), **common)
    val_ds = build_caption_dataset(
        dcfg["loader"], split="val",
        max_samples=dcfg.get("max_val_samples"),
        caption_selection="first", **common)
    print(f"[data] loader={dcfg['loader']} train={len(train_ds)} "
          f"val={len(val_ds)}")

    # All training captions (every reference, not just the selected one).
    train_captions: list[str] = []
    for i in range(len(train_ds)):
        item = train_ds[i]
        train_captions.extend(getattr(item, "references", None) or [item.caption])

    refs: list[list[str]] = []
    filenames: list[str] = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        refs.append(list(getattr(item, "references", None) or [item.caption]))
        filenames.append(item.filename)
    image_paths = [images_dir / f for f in filenames]

    predictions = build_baselines(train_captions, len(val_ds), cfg["seed"])
    for name, hyps in predictions.items():
        print(f"[baseline] {name}: {hyps[0][:80]!r}")

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps({"filenames": filenames, "references": refs,
                    "predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mcfg = cfg["metrics"]
    metrics: dict[str, dict[str, float]] = {}
    for name, hyps in predictions.items():
        print(f"[score] {name}")
        row: dict[str, float] = {}
        row["mean_len_words"] = sum(len(h.split()) for h in hyps) / len(hyps)

        ng = ngram_scores(hyps, refs)
        row.update(bleu1=ng.bleu1, bleu2=ng.bleu2, bleu3=ng.bleu3,
                   bleu4=ng.bleu4, cider=ng.cider)

        if mcfg.get("bertscore", True):
            row.update(bertscore_bn(
                hyps, refs, device=str(device),
                batch_size=mcfg.get("bertscore_batch_size", 64)))
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
                    batch_size=mcfg.get("clipscore_batch_size", 32))
            except ImportError:
                print("  [warn] sentence-transformers missing")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metrics[name] = row
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = list(next(iter(metrics.values())).keys())
    md = [
        f"# Trivial no-vision baselines — `{dcfg['loader']}` val split",
        "",
        f"{len(val_ds)} images, multi-reference scoring "
        f"(seed={cfg['seed']}). Baselines derived from the train split "
        "only; none of them look at the image.",
        "",
        "| Baseline | " + " | ".join(cols) + " |",
        "|---|" + "|".join("---:" for _ in cols) + "|",
    ]
    for name, row in metrics.items():
        md.append("| `" + name + "` | "
                  + " | ".join(f"{row.get(c, float('nan')):.4f}" for c in cols)
                  + " |")
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote: {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
