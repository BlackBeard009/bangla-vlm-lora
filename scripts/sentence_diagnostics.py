"""Post-hoc diagnostics on a saved bridged-GiT + LoRA checkpoint.

Tries to answer: *why* does the model emit short, templated captions?

Runs three independent diagnostics on the same adapter, no retraining:

  - Test 1 (image_scrambling): does the visual signal actually reach the
    decoder? Compares outputs across real / gray / noise / shuffled
    image inputs with baseline_beam4 decode.

  - Test 2 (forced_length): does the model have more compositional
    structure than beam search reveals? Compares outputs across beam4
    decode configs with progressively stronger min_new_tokens.

  - Test 3 (caption_length_eval): how badly does the model underfit the
    LONG captions in BanglaLekha? Computes val cross-entropy under
    caption_selection in {first, last, random}; captions[0] is the
    ~7-word short form and captions[-1] is the ~10+ word descriptive
    form.

Run from repo root:
    python scripts/sentence_diagnostics.py --config configs/sentence_diagnostics.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from src.data.banglalekha import BanglaLekhaCaptions  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bridged_lora(base_name: str, adapter_dir: Path, dtype: torch.dtype, device: torch.device):
    """Reconstruct the bridged-GiT + LoRA bundle from a saved checkpoint.

    Mirrors scripts/decode_ablation.py::load_bridged_lora. Lift into
    src/models/ once a third script needs it.
    """
    processor = AutoProcessor.from_pretrained(base_name)
    bridged_tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    processor.tokenizer = bridged_tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    base_model.resize_token_embeddings(len(bridged_tokenizer))

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.to(device)
    peft_model.eval()
    return peft_model, processor


# ---------- Test 1 helpers ----------

def make_perturbed_image(real: Image.Image, mode: str, rng: np.random.Generator) -> Image.Image:
    """Build a perturbed image at the same size as `real`.

    Modes: 'gray' (solid (128,128,128)), 'noise' (per-pixel uniform RGB).
    Returns a PIL image; the processor will normalize it.
    """
    w, h = real.size
    if mode == "gray":
        return Image.new("RGB", (w, h), color=(128, 128, 128))
    if mode == "noise":
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"unknown perturbation mode: {mode!r}")


@torch.no_grad()
def decode_one(model, processor, image: Image.Image, params: dict, device) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    gen_ids = model.generate(pixel_values=pixel_values, **params)
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


def run_image_scrambling(model, processor, items, decode_params, variants, device, seed):
    """Run all image-scrambling variants on the same 8 val items.

    Returns dict {variant_name: [{filename, reference, generated}]}.
    """
    rng = np.random.default_rng(seed)
    results: dict[str, list[dict]] = {v["name"]: [] for v in variants}
    n = len(items)
    for i, item in enumerate(items):
        per_image = {}
        for variant in variants:
            mode = variant["name"]
            if mode == "real":
                image = item.image
            elif mode == "shuffled":
                # Use the next image in the batch as input
                image = items[(i + 1) % n].image
            else:
                image = make_perturbed_image(item.image, mode, rng)
            decoded = decode_one(model, processor, image, decode_params, device)
            per_image[mode] = decoded
        for mode, gen in per_image.items():
            results[mode].append({
                "filename": item.filename,
                "reference": item.caption,
                "generated": gen,
            })
    return results


def overlap_metric(a: str, b: str) -> float:
    """Token-level Jaccard between two strings (whitespace tokenization).

    1.0 means identical; 0.0 means no shared tokens.
    """
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def scrambling_summary(results: dict[str, list[dict]], variants) -> dict:
    """For each non-'real' variant: % exact-match-with-real + mean Jaccard."""
    real = {r["filename"]: r["generated"] for r in results["real"]}
    summary = {}
    for v in variants:
        name = v["name"]
        if name == "real":
            continue
        exact = 0
        jaccards = []
        n = 0
        for r in results[name]:
            real_gen = real[r["filename"]]
            if real_gen == r["generated"]:
                exact += 1
            jaccards.append(overlap_metric(real_gen, r["generated"]))
            n += 1
        summary[name] = {
            "exact_match_pct": 100.0 * exact / max(n, 1),
            "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
            "n": n,
        }
    return summary


# ---------- Test 2 helpers ----------

def run_forced_length(model, processor, items, variants, device, seed):
    results: dict[str, list[dict]] = {}
    for variant in variants:
        set_seed(seed)
        name = variant["name"]
        params = dict(variant["params"])
        out = []
        for item in items:
            decoded = decode_one(model, processor, item.image, params, device)
            out.append({
                "filename": item.filename,
                "reference": item.caption,
                "generated": decoded,
                "ref_tokens": len(item.caption.split()),
                "gen_tokens": len(decoded.split()),
            })
        results[name] = out
    return results


def forced_length_summary(results):
    summary = {}
    for name, out in results.items():
        gen_lens = [r["gen_tokens"] for r in out]
        ref_lens = [r["ref_tokens"] for r in out]
        summary[name] = {
            "mean_gen_len": float(np.mean(gen_lens)),
            "mean_ref_len": float(np.mean(ref_lens)),
            "min_gen_len": int(np.min(gen_lens)),
            "max_gen_len": int(np.max(gen_lens)),
        }
    return summary


# ---------- Test 3 helpers ----------

def collate_loss(batch, processor):
    images = [item.image for item in batch]
    captions = [item.caption for item in batch]
    enc = processor(images=images, text=captions, padding="longest", return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


@torch.no_grad()
def val_loss_for_selection(model, processor, captions_path, images_dir, val_fraction, batch_size, selection, seed, device):
    val_ds = BanglaLekhaCaptions(
        captions_path=captions_path,
        images_dir=images_dir,
        split="val",
        val_fraction=val_fraction,
        caption_selection=selection,
        seed=seed,
    )
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_loss(b, processor),
    )
    total_loss = 0.0
    n_batches = 0
    # Also track caption-length stats for the summary
    cap_word_lens = []
    for item in val_ds.entries:
        if selection == "first":
            cap = item["caption"][0]
        elif selection == "last":
            cap = item["caption"][-1]
        else:
            cap = item["caption"][0]  # not the actual sampled one, but a representative
        cap_word_lens.append(len(cap.split()))
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
    return {
        "selection": selection,
        "n_val": len(val_ds),
        "val_loss": total_loss / max(n_batches, 1),
        "perplexity": float(np.exp(total_loss / max(n_batches, 1))),
        "mean_ref_words": float(np.mean(cap_word_lens)),
    }


# ---------- Reporting ----------

def write_report(cfg, out_dir: Path, t1_results, t1_summary, t2_results, t2_summary, t3_results, n_val):
    lines = [
        "# Sentence-length diagnostics — banglalekha_full adapter",
        "",
        f"Adapter: `paths.EXP_CHECKPOINTS/{cfg['adapter']['checkpoint_subdir']}/`",
        f"Val split: {n_val} images (seed={cfg['seed']}, val_fraction={cfg['data']['val_fraction']}).",
        "",
        "Three independent diagnostics; none touch the model weights.",
        "Aim is to disambiguate the cause(s) of the templated outputs",
        "observed in `banglalekha_full/README.md`.",
        "",
    ]

    # ---- Test 1
    if cfg.get("image_scrambling", {}).get("enabled"):
        lines += [
            "## Test 1 — image scrambling",
            "",
            "Same 8 val items, identical decode params "
            f"(`{cfg['image_scrambling']['decode']}`), four image inputs:",
            "",
        ]
        for v in cfg["image_scrambling"]["variants"]:
            lines.append(f"- `{v['name']}` — {v['description']}")
        lines += [
            "",
            "### Outputs",
            "",
            "| Image | Reference |" + "".join(f" `{v['name']}` |" for v in cfg["image_scrambling"]["variants"]),
            "|---|---|" + "|".join(["---"] * len(cfg["image_scrambling"]["variants"])) + "|",
        ]
        n = len(t1_results["real"])
        for i in range(n):
            fname = t1_results["real"][i]["filename"]
            ref = t1_results["real"][i]["reference"]
            row = [f"`{fname}`", ref]
            for v in cfg["image_scrambling"]["variants"]:
                row.append(t1_results[v["name"]][i]["generated"])
            lines.append("| " + " | ".join(row) + " |")
        lines += [
            "",
            "### Divergence from `real` baseline",
            "",
            "| Variant | Exact-match with `real` | Mean Jaccard token overlap |",
            "|---|---:|---:|",
        ]
        for v in cfg["image_scrambling"]["variants"]:
            if v["name"] == "real":
                continue
            s = t1_summary[v["name"]]
            lines.append(f"| `{v['name']}` | {s['exact_match_pct']:.1f}% | {s['mean_jaccard']:.3f} |")
        lines += [
            "",
            "**Read:** exact-match of 100% / Jaccard of 1.0 means the model",
            "produced an identical caption with a degraded image — i.e. it",
            "ignored visual input. Lower numbers mean the visual signal is",
            "shaping outputs.",
            "",
        ]

    # ---- Test 2
    if cfg.get("forced_length", {}).get("enabled"):
        lines += [
            "## Test 2 — forced-length decode",
            "",
            "Same 8 val items, real images. Decode hyperparameters vary:",
            "",
        ]
        for v in cfg["forced_length"]["variants"]:
            lines.append(f"- `{v['name']}` — {v['description']}")
        lines += [
            "",
            "### Outputs",
            "",
            "| Image | Reference |" + "".join(f" `{v['name']}` |" for v in cfg["forced_length"]["variants"]),
            "|---|---|" + "|".join(["---"] * len(cfg["forced_length"]["variants"])) + "|",
        ]
        n = len(t2_results["baseline_beam4"])
        for i in range(n):
            fname = t2_results["baseline_beam4"][i]["filename"]
            ref = t2_results["baseline_beam4"][i]["reference"]
            row = [f"`{fname}`", ref]
            for v in cfg["forced_length"]["variants"]:
                row.append(t2_results[v["name"]][i]["generated"])
            lines.append("| " + " | ".join(row) + " |")
        lines += [
            "",
            "### Length summary (words by whitespace)",
            "",
            "| Variant | Mean gen len | Min | Max | Mean ref len |",
            "|---|---:|---:|---:|---:|",
        ]
        for v in cfg["forced_length"]["variants"]:
            s = t2_summary[v["name"]]
            lines.append(
                f"| `{v['name']}` | {s['mean_gen_len']:.1f} | "
                f"{s['min_gen_len']} | {s['max_gen_len']} | {s['mean_ref_len']:.1f} |"
            )
        lines.append("")

    # ---- Test 3
    if cfg.get("caption_length_eval", {}).get("enabled"):
        lines += [
            "## Test 3 — caption-length val cross-entropy",
            "",
            "Full 915-item val split. Same adapter, same decode-irrelevant",
            "teacher-forced forward pass; only the reference target changes.",
            "",
            "| Selection | Mean ref words | Val loss | Perplexity | n_val |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in t3_results:
            lines.append(
                f"| `{r['selection']}` | {r['mean_ref_words']:.1f} | "
                f"{r['val_loss']:.3f} | {r['perplexity']:.1f} | {r['n_val']} |"
            )
        if len(t3_results) >= 2:
            short = next((r for r in t3_results if r["selection"] == "first"), None)
            long_ = next((r for r in t3_results if r["selection"] == "last"), None)
            if short and long_:
                delta = long_["val_loss"] - short["val_loss"]
                lines += [
                    "",
                    f"**Gap (`last` − `first`):** {delta:+.3f} CE / "
                    f"{long_['perplexity']/short['perplexity']:.2f}× perplexity.",
                ]
        lines.append("")

    lines += [
        "## Artifacts",
        "",
        f"- `paths.EXP_RESULTS/{cfg['output']['results_subdir']}/results.md` — this report.",
        f"- `paths.EXP_RESULTS/{cfg['output']['results_subdir']}/samples.json` — all generations + per-image inputs.",
        f"- `paths.EXP_RESULTS/{cfg['output']['results_subdir']}/summary.json` — numeric summaries.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /content/bangla-vlm-lora",
        "python scripts/sentence_diagnostics.py --config configs/sentence_diagnostics.yaml",
        "```",
        "",
        f"Requires the `{cfg['adapter']['checkpoint_subdir']}` adapter on Drive and",
        "BanglaLekha images extracted to `images_dir` in the config (see the",
        "Reproduce block in `docs/results/banglalekha_full/README.md`).",
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[cfg["model"]["dtype"]]

    adapter_dir = paths.EXP_CHECKPOINTS / cfg["adapter"]["checkpoint_subdir"]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")
    print(f"[adapter] {adapter_dir}")

    dcfg = cfg["data"]
    val_ds_first = BanglaLekhaCaptions(
        captions_path=dcfg["captions_path"],
        images_dir=dcfg["images_dir"],
        split="val",
        val_fraction=dcfg["val_fraction"],
        caption_selection="first",
        seed=cfg["seed"],
    )
    n = min(dcfg["n_samples"], len(val_ds_first))
    items = [val_ds_first[i] for i in range(n)]
    print(f"[data] val={len(val_ds_first)}, decoding first {n} items for tests 1+2")

    print(f"[model] loading {cfg['model']['name']} + bridge + adapter ...")
    t0 = time.time()
    model, processor = load_bridged_lora(cfg["model"]["name"], adapter_dir, dtype, device)
    print(f"  vocab_size={len(processor.tokenizer)}  load={time.time()-t0:.1f}s")

    samples_dump = {}
    summary_dump = {}

    # ---- Test 1
    t1_summary = None
    t1_results = None
    if cfg.get("image_scrambling", {}).get("enabled"):
        print("\n[test 1] image_scrambling ...")
        t0 = time.time()
        t1_results = run_image_scrambling(
            model, processor, items,
            decode_params=cfg["image_scrambling"]["decode"],
            variants=cfg["image_scrambling"]["variants"],
            device=device,
            seed=cfg["seed"],
        )
        t1_summary = scrambling_summary(t1_results, cfg["image_scrambling"]["variants"])
        print(f"  done in {time.time()-t0:.1f}s")
        for v in cfg["image_scrambling"]["variants"]:
            if v["name"] == "real":
                continue
            s = t1_summary[v["name"]]
            print(f"    {v['name']:>10}: exact={s['exact_match_pct']:.1f}%  jaccard={s['mean_jaccard']:.3f}")
        samples_dump["image_scrambling"] = t1_results
        summary_dump["image_scrambling"] = t1_summary

    # ---- Test 2
    t2_summary = None
    t2_results = None
    if cfg.get("forced_length", {}).get("enabled"):
        print("\n[test 2] forced_length ...")
        t0 = time.time()
        t2_results = run_forced_length(
            model, processor, items,
            variants=cfg["forced_length"]["variants"],
            device=device,
            seed=cfg["seed"],
        )
        t2_summary = forced_length_summary(t2_results)
        print(f"  done in {time.time()-t0:.1f}s")
        for name, s in t2_summary.items():
            print(f"    {name:>22}: mean_gen={s['mean_gen_len']:.1f} (min={s['min_gen_len']}, max={s['max_gen_len']})")
        samples_dump["forced_length"] = t2_results
        summary_dump["forced_length"] = t2_summary

    # ---- Test 3
    t3_results = None
    if cfg.get("caption_length_eval", {}).get("enabled"):
        print("\n[test 3] caption_length_eval ...")
        t3_results = []
        for sel in cfg["caption_length_eval"]["selections"]:
            t0 = time.time()
            r = val_loss_for_selection(
                model, processor,
                captions_path=dcfg["captions_path"],
                images_dir=dcfg["images_dir"],
                val_fraction=dcfg["val_fraction"],
                batch_size=cfg["caption_length_eval"]["batch_size"],
                selection=sel,
                seed=cfg["seed"],
                device=device,
            )
            print(f"    {sel:>7}: val_loss={r['val_loss']:.3f}  ppl={r['perplexity']:.1f}  "
                  f"mean_ref_words={r['mean_ref_words']:.1f}  ({time.time()-t0:.1f}s)")
            t3_results.append(r)
        summary_dump["caption_length_eval"] = t3_results

    # Persist
    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.json").write_text(
        json.dumps(samples_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(cfg, out_dir, t1_results, t1_summary, t2_results, t2_summary, t3_results, len(val_ds_first))
    print(f"\nWrote: {out_dir / 'results.md'}")
    print(f"       {out_dir / 'samples.json'}")
    print(f"       {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
