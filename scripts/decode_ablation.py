"""Decode-side ablation on a saved bridged-GiT + LoRA checkpoint.

Loads the adapter + bridged tokenizer from
``paths.EXP_CHECKPOINTS / cfg.adapter.checkpoint_subdir``, builds a val
dataset with the same seed used at train time, and runs generation on the
first ``data.n_samples`` items with each of the variants in ``cfg.variants``.

No training. Pure decode comparison — meant to test whether output
quality (length, repetition, lexical variety) responds to decode
hyperparameters before we invest in more training or architectural
changes.

Run from repo root:
    python scripts/decode_ablation.py --config configs/decode_ablation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from src.data.banglalekha import BanglaLekhaCaptions  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bridged_lora(
    base_name: str,
    adapter_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
):
    """Reconstruct the bridged-GiT + LoRA bundle from a saved checkpoint.

    The checkpoint contains:
      - adapter_model.safetensors  (LoRA deltas + modules_to_save weights)
      - adapter_config.json
      - tokenizer.json + tokenizer_config.json  (bridged tokenizer)

    We restore by (1) loading the base GiT, (2) restoring the bridged
    tokenizer, (3) resizing the base model's embeddings to match the
    bridged vocab, (4) loading the PEFT adapter (which restores both LoRA
    deltas and the trained word_embeddings / output via modules_to_save).
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


@torch.no_grad()
def generate_variant(model, processor, items, params: dict, device) -> list[dict]:
    out = []
    for item in items:
        pixel_values = processor(
            images=item.image, return_tensors="pt"
        ).pixel_values.to(device)
        gen_ids = model.generate(pixel_values=pixel_values, **params)
        decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        out.append({
            "filename": item.filename,
            "reference": item.caption,
            "generated": decoded,
        })
    return out


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

    print("[data] loading val split ...")
    dcfg = cfg["data"]
    val_ds = BanglaLekhaCaptions(
        captions_path=dcfg["captions_path"],
        images_dir=dcfg["images_dir"],
        split="val",
        val_fraction=dcfg["val_fraction"],
        caption_selection=dcfg.get("caption_selection", "first"),
        seed=cfg["seed"],
    )
    n = min(dcfg["n_samples"], len(val_ds))
    items = [val_ds[i] for i in range(n)]
    print(f"  val={len(val_ds)}, decoding first {n} items")

    print(f"[model] loading {cfg['model']['name']} + bridge + adapter ...")
    model, processor = load_bridged_lora(cfg["model"]["name"], adapter_dir, dtype, device)
    print(f"  vocab_size={len(processor.tokenizer)}")

    all_results = {}
    for variant in cfg["variants"]:
        # Re-seed before sampling variants so do_sample is reproducible.
        set_seed(cfg["seed"])
        name = variant["name"]
        params = dict(variant["params"])
        print(f"[decode] variant={name}  params={params}")
        all_results[name] = generate_variant(model, processor, items, params, device)
        for s in all_results[name][:3]:
            print(f"  {s['filename']}: {s['generated']!r}")

    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / "samples.json"
    samples_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Decode-side ablation — banglalekha_full adapter",
        "",
        f"Adapter: `paths.EXP_CHECKPOINTS/{cfg['adapter']['checkpoint_subdir']}/`",
        f"Val items decoded: first {n} of {len(val_ds)} (seed={cfg['seed']}, "
        f"caption_selection={dcfg.get('caption_selection', 'first')!r}).",
        "",
        "## Variant settings",
        "",
        "| Variant | Params |",
        "|---|---|",
    ]
    for variant in cfg["variants"]:
        md_lines.append(f"| `{variant['name']}` | `{variant['params']}` |")

    md_lines += ["", "## Sample comparisons", ""]
    md_lines.append("| Image | Reference |" + " |".join(f" `{v['name']}`" for v in cfg["variants"]) + " |")
    md_lines.append("|---|---|" + "|".join("---" for _ in cfg["variants"]) + "|")
    for i in range(n):
        ref = items[i].caption
        fname = items[i].filename
        gens = []
        for variant in cfg["variants"]:
            g = all_results[variant["name"]][i]["generated"]
            gens.append(g)
        md_lines.append("| `" + fname + "` | " + ref + " | " + " | ".join(gens) + " |")

    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nWrote: {md_path}\n       {samples_path}")


if __name__ == "__main__":
    main()
