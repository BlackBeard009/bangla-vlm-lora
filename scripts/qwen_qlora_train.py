"""QLoRA fine-tuning of Qwen2-VL-2B-Instruct on Bangla captions.

The contrast experiment to the GiT bridge chain: Qwen2-VL's tokenizer
already covers Bangla, yet the instruct model produces zero Bangla
zero-shot (see docs/results/bancap_metrics/README.md). This run asks:
does plain LoRA on the LLM attention projections unlock the Bangla the
vocabulary already covers — no tokenizer surgery, no embedding bridge?

Recipe: 4-bit NF4 base (bitsandbytes), LoRA r=8 on q/k/v/o_proj of the
language model only (vision tower + embeddings + LM head frozen —
deliberately mirrors the GiT attention-only V1 ablation, which FAILED
for GiT because the vocab was missing; if it SUCCEEDS here, vocabulary
coverage is confirmed as the deciding factor).

Training format = the zero-shot prompt with the reference caption as
the assistant turn; prompt tokens are masked from the loss. Eval-time
generation uses the identical prompt, so the zero-shot row is the exact
ablation baseline.

Run:
    python scripts/qwen_qlora_train.py --config configs/qwen_qlora_bancap.yaml
"""

from __future__ import annotations

import argparse
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


def build_texts(processor, prompt: str, caption: str | None) -> str:
    """Chat-template text; with caption -> full training text, without ->
    generation prompt."""
    user = {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }
    if caption is None:
        return processor.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True
        )
    assistant = {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    return processor.apply_chat_template([user, assistant], tokenize=False)


def encode_sample(processor, item, prompt: str, device):
    """Return model inputs with labels masked over the prompt span."""
    text_full = build_texts(processor, prompt, item.caption)
    text_prompt = build_texts(processor, prompt, None)
    enc = processor(text=[text_full], images=[item.image], return_tensors="pt")
    enc_prompt = processor(text=[text_prompt], images=[item.image],
                           return_tensors="pt")
    n_prompt = enc_prompt["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def val_ce(model, processor, val_ds, n_items: int, prompt: str, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(min(n_items, len(val_ds))):
        batch = encode_sample(processor, val_ds[i], prompt, device)
        total += model(**batch).loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def sample_generations(model, processor, val_ds, n_samples: int, prompt: str,
                       gen_cfg: dict, device) -> list[dict]:
    model.eval()
    out = []
    text_prompt = build_texts(processor, prompt, None)
    for i in range(min(n_samples, len(val_ds))):
        item = val_ds[i]
        inputs = processor(text=[text_prompt], images=[item.image],
                           return_tensors="pt").to(device)
        gen = model.generate(**inputs, **gen_cfg)
        dec = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        out.append({"filename": item.filename, "reference": item.caption,
                    "generated": dec})
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    set_seed(cfg["seed"])
    device = torch.device("cuda")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig)

    dcfg = cfg["data"]
    common = dict(
        captions_path=resolve_path(dcfg["captions_path"]),
        images_dir=resolve_path(dcfg["images_dir"]),
        val_fraction=dcfg["val_fraction"],
        seed=cfg["seed"],
    )
    train_ds = build_caption_dataset(
        dcfg["loader"], split="train",
        max_samples=dcfg.get("max_samples"),
        caption_selection=dcfg.get("caption_selection", "random"), **common)
    val_ds = build_caption_dataset(
        dcfg["loader"], split="val",
        max_samples=dcfg.get("max_val_samples"),
        caption_selection="first", **common)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    name = cfg["model"]["name"]
    print(f"[model] {name} 4-bit NF4 + LoRA ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor_kwargs = cfg["model"].get("processor_kwargs") or {}
    processor = AutoProcessor.from_pretrained(name, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        name, quantization_config=bnb, dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)

    lcfg = cfg["lora"]
    init_subdir = lcfg.get("init_adapter_subdir")
    if init_subdir:
        # Curriculum stage 2+: continue training an existing adapter
        # (e.g. BanglaView silver pretrain -> BAN-Cap gold finetune).
        from peft import PeftModel
        init_dir = paths.EXP_CHECKPOINTS / init_subdir
        print(f"[lora] resuming adapter from {init_dir}")
        model = PeftModel.from_pretrained(model, init_dir, is_trainable=True)
    else:
        lora = LoraConfig(
            r=lcfg["rank"], lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            bias="none", target_modules=list(lcfg["target_modules"]),
            # No task_type on purpose — same reasoning as the GiT wiring.
        )
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    import bitsandbytes as bnb_optim
    tcfg = cfg["training"]
    optim = bnb_optim.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    accum = tcfg["grad_accum"]
    prompt = cfg["prompt"]
    gen_cfg = dict(cfg["generation"])

    model.train()
    eval_points = []
    order = torch.randperm(len(train_ds)).tolist()
    cursor = 0
    t0 = time.time()
    window: list[float] = []
    for step in range(tcfg["steps"]):
        optim.zero_grad(set_to_none=True)
        for _ in range(accum):
            if cursor >= len(order):
                order = torch.randperm(len(train_ds)).tolist()
                cursor = 0
            item = train_ds[order[cursor]]
            cursor += 1
            batch = encode_sample(processor, item, prompt, device)
            loss = model(**batch).loss / accum
            loss.backward()
            window.append(loss.item() * accum)
        optim.step()

        if step % tcfg["log_every"] == 0:
            mean = sum(window[-accum:]) / accum
            print(f"  step {step:>5}  loss={mean:.4f}  "
                  f"({(time.time() - t0):.0f}s)")

        if step > 0 and step % tcfg["eval_every"] == 0:
            ce = val_ce(model, processor, val_ds, tcfg["val_ce_items"],
                        prompt, device)
            samples = sample_generations(model, processor, val_ds, 4,
                                         prompt, gen_cfg, device)
            eval_points.append({
                "step": step,
                "train_loss": sum(window) / max(len(window), 1),
                "val_ce": ce,
                "samples": samples,
            })
            window.clear()
            print(f"  [eval] step {step}: val_ce={ce:.4f}")
            for s in samples[:2]:
                print(f"    {s['filename']}: {s['generated']!r}")

    ce = val_ce(model, processor, val_ds, tcfg["val_ce_items"], prompt, device)
    samples = sample_generations(model, processor, val_ds, 8, prompt,
                                 gen_cfg, device)
    wall = time.time() - t0
    eval_points.append({
        "step": tcfg["steps"],
        "train_loss": sum(window) / max(len(window), 1) if window else None,
        "val_ce": ce,
        "samples": samples,
    })

    print("\n" + "=" * 70)
    print(f"FINAL  val_ce={ce:.4f}  wall={wall:.1f}s")
    print("=" * 70)
    for s in samples:
        print(f"  {s['filename']}\n    ref: {s['reference']}\n    gen: {s['generated']}")

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Qwen2-VL-2B QLoRA — `{cfg['output']['results_subdir']}`",
        "",
        f"Base: `{name}` (4-bit NF4, bf16 compute)",
        f"LoRA: r={lcfg['rank']} on `{lcfg['target_modules']}` "
        f"(LLM only; vision/embeddings/head frozen)",
        f"Trainable: {trainable:,} ({100 * trainable / total:.3f}%)",
        f"Steps: {tcfg['steps']} (batch 1 x accum {accum}, lr {tcfg['lr']})",
        f"Prompt: `{prompt}`",
        f"Wall: {wall:.1f}s",
        "",
        "| Step | Train loss | Val CE |",
        "|---:|---:|---:|",
    ]
    for ep in eval_points:
        tl = f"{ep['train_loss']:.3f}" if ep["train_loss"] else "-"
        lines.append(f"| {ep['step']} | {tl} | {ep['val_ce']:.3f} |")
    lines += ["", "## Final sample generations", ""]
    for s in samples:
        lines += [f"**{s['filename']}**",
                  f"- Reference: `{s['reference']}`",
                  f"- Generated: `{s['generated']}`", ""]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    (out_dir / "samples.json").write_text(
        json.dumps([ep["samples"] for ep in eval_points],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote: {out_dir / 'results.md'}")

    if cfg["output"].get("save_adapter"):
        ckpt = paths.EXP_CHECKPOINTS / cfg["output"]["results_subdir"]
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        print(f"       {ckpt} (adapter + processor)")


if __name__ == "__main__":
    main()
