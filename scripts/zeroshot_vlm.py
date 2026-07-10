"""Zero-shot multilingual-VLM captioning baseline (paper-plan §C4).

Prompts an instruction-tuned VLM (Qwen2-VL-2B-Instruct by default) for a
one-sentence Bangla caption on every image of a corpus val split, then
scores the outputs with the same metric battery as
``scripts/score_captions.py`` (multi-ref BLEU-1..4 / CIDEr /
BERTScore-bn / M-CLIPScore) so rows are directly comparable.

Establishes the zero-shot lower bound that the LoRA-adapted models must
beat, and doubles as the "tokenizer already has Bangla" contrast system
for the embedding-bridge story.

Run:
    python scripts/zeroshot_vlm.py --config configs/zeroshot_qwen2vl_bancap.yaml
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


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


@torch.no_grad()
def generate_zero_shot(model, processor, dataset, prompt: str,
                       gen_params: dict, device) -> list[str]:
    hyps: list[str] = []
    n = len(dataset)
    t0 = time.time()
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    for i in range(n):
        item = dataset[i]
        inputs = processor(
            text=[chat_text], images=[item.image], return_tensors="pt"
        ).to(device)
        out = model.generate(**inputs, **gen_params)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()
        hyps.append(decoded)
        if i % 50 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"    {i + 1}/{n} ({rate:.2f} img/s)  last={decoded[:60]!r}")
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[cfg["model"]["dtype"]]

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
        raise RuntimeError(f"val split empty — check images_dir={images_dir}")
    print(f"[data] loader={dcfg['loader']}  val={len(val_ds)}")

    refs: list[list[str]] = []
    filenames: list[str] = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        refs.append(list(getattr(item, "references", None) or [item.caption]))
        filenames.append(item.filename)
    image_paths = [images_dir / f for f in filenames]

    # ---- Phase 1: zero-shot generation --------------------------------
    from transformers import AutoModelForImageTextToText, AutoProcessor

    name = cfg["model"]["name"]
    print(f"[model] loading {name} ({cfg['model']['dtype']}) ...")
    processor_kwargs = cfg["model"].get("processor_kwargs") or {}
    processor = AutoProcessor.from_pretrained(name, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()

    prompt = cfg["prompt"]
    gen_params = dict(cfg["generation"])
    print(f"[generate] prompt={prompt!r}  params={gen_params}")
    hyps = generate_zero_shot(model, processor, val_ds, prompt,
                              gen_params, device)

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps(
            {"filenames": filenames, "references": refs,
             "predictions": {"zero_shot": hyps}},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # ---- Phase 2: metrics ---------------------------------------------
    mcfg = cfg["metrics"]
    row: dict[str, float] = {}
    row["mean_len_words"] = sum(len(h.split()) for h in hyps) / len(hyps)
    n_empty = sum(1 for h in hyps if not h.strip())
    row["n_empty"] = float(n_empty)

    ng = ngram_scores(hyps, refs)
    row.update(bleu1=ng.bleu1, bleu2=ng.bleu2, bleu3=ng.bleu3,
               bleu4=ng.bleu4, cider=ng.cider)

    if mcfg.get("bertscore", True):
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
            print("  [warn] sentence-transformers missing — no M-CLIPScore")

    metrics = {"zero_shot": row}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cols = list(row.keys())
    md = [
        f"# Zero-shot captioning baseline — `{name}`",
        "",
        f"Corpus: `{dcfg['loader']}` val split, {len(val_ds)} images, "
        f"multi-reference scoring (seed={cfg['seed']}).",
        f"Prompt: `{prompt}`",
        f"Generation: `{gen_params}`",
        "",
        "| System | " + " | ".join(cols) + " |",
        "|---|" + "|".join("---:" for _ in cols) + "|",
        f"| `{name}` zero-shot | "
        + " | ".join(f"{row[c]:.4f}" for c in cols) + " |",
    ]
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n" + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))
    print(f"\nWrote: {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
