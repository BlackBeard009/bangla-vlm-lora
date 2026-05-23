"""Bridged-GiT + LoRA training on BAN-Cap.

Trains the bridged GiT-base + LoRA recipe on BAN-Cap (8,091 Flickr8k
images × 5 native Bengali captions per image). Same hyperparameters
as `scripts/banglalekha_slice.py` so the two runs are directly
comparable; the only meaningful difference is the dataset.

Mirrors `scripts/banglalekha_slice.py` structurally — TODO: lift the
shared training/eval/report machinery into `src/training/` when a third
corpus loader lands.

Run from repo root:
    python scripts/bancap_train.py --config configs/bancap_slice.yaml
    python scripts/bancap_train.py --config configs/bancap_full.yaml
"""

from __future__ import annotations

import argparse
import json
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
from src.data.bancap import BanCapCaptions  # noqa: E402
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
class EvalPoint:
    step: int
    train_loss: float       # mean of the most-recent log window
    val_loss: float
    samples: list[dict] = field(default_factory=list)  # {filename, reference, generated}


@dataclass
class RunResult:
    init_strategy: str
    vocab_size_before: int
    vocab_size_after: int
    n_added: int
    trainable: int
    trainable_pct: float
    n_train: int
    n_val: int
    wall_seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    eval_points: list[EvalPoint] = field(default_factory=list)


@torch.no_grad()
def evaluate(bundle, val_loader, device) -> float:
    was_training = bundle.model.training
    bundle.model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = bundle.model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
    if was_training:
        bundle.model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def sample_generations(bundle, val_ds, n_samples, gen_cfg, device) -> list[dict]:
    was_training = bundle.model.training
    bundle.model.eval()
    samples = []
    for i in range(min(n_samples, len(val_ds))):
        item = val_ds[i]
        pixel_values = bundle.processor(
            images=item.image, return_tensors="pt"
        ).pixel_values.to(device)
        gen_ids = bundle.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=gen_cfg["max_new_tokens"],
            num_beams=gen_cfg["num_beams"],
        )
        decoded = bundle.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        samples.append({
            "filename": item.filename,
            "reference": item.caption,
            "generated": decoded,
        })
    if was_training:
        bundle.model.train()
    return samples


def write_report(result: RunResult, out_path: Path, cfg: dict) -> None:
    lines = [
        "# BAN-Cap run — bridged GiT + LoRA",
        "",
        f"Config: `{cfg['output']['results_subdir']}`",
        f"Init strategy: `{result.init_strategy}`",
        f"Vocab: {result.vocab_size_before} → {result.vocab_size_after} (+{result.n_added})",
        f"Trainable: {result.trainable:,} ({result.trainable_pct:.2f}%)",
        f"Train / val: {result.n_train} / {result.n_val}",
        f"Steps: {cfg['training']['steps']} (batch {cfg['training']['batch_size']}, lr {cfg['training']['lr']})",
        f"Caption selection (train): `{cfg['data'].get('caption_selection', 'first')}`",
        f"Wall: {result.wall_seconds:.1f}s",
        "",
        "## Evaluation points",
        "",
        "| Step | Train loss | Val loss |",
        "|---:|---:|---:|",
    ]
    for ep in result.eval_points:
        lines.append(f"| {ep.step} | {ep.train_loss:.3f} | {ep.val_loss:.3f} |")

    if result.eval_points:
        lines += ["", "## Sample generations (final eval point)", ""]
        for s in result.eval_points[-1].samples:
            lines += [
                f"**{s['filename']}**",
                f"- Reference: `{s['reference']}`",
                f"- Generated: `{s['generated']}`",
                "",
            ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    print("[data] loading BAN-Cap ...")
    dcfg = cfg["data"]
    train_ds = BanCapCaptions(
        captions_path=dcfg["captions_path"],
        images_dir=dcfg["images_dir"],
        split="train",
        val_fraction=dcfg["val_fraction"],
        max_samples=dcfg.get("max_samples"),
        caption_selection=dcfg.get("caption_selection", "first"),
        seed=cfg["seed"],
    )
    val_ds = BanCapCaptions(
        captions_path=dcfg["captions_path"],
        images_dir=dcfg["images_dir"],
        split="val",
        val_fraction=dcfg["val_fraction"],
        max_samples=dcfg.get("max_val_samples"),
        caption_selection="first",
        seed=cfg["seed"],
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}")

    print("[model] loading GiT-base ...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    bundle = load_git(
        cfg["model"]["name"],
        dtype=dtype_map[cfg["model"]["dtype"]],
        use_fast_tokenizer=cfg["model"].get("use_fast_tokenizer", True),
    )

    print(f"[bridge] init_strategy={cfg['bridge']['init_strategy']}")
    report = bridge_vocabulary(
        bundle,
        donor_tokenizer=cfg["bridge"]["donor_tokenizer"],
        donor_model_for_init=cfg["bridge"]["donor_model_for_init"],
        init_strategy=cfg["bridge"]["init_strategy"],
    )
    print(f"  vocab {report.n_existing} -> {report.n_final} (+{report.n_added})")

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
    print(f"[lora] trainable {p['trainable']:,} / {p['total']:,} ({p['trainable_pct']:.2f}%)")

    result = RunResult(
        init_strategy=cfg["bridge"]["init_strategy"],
        vocab_size_before=report.n_existing,
        vocab_size_after=report.n_final,
        n_added=report.n_added,
        trainable=p["trainable"],
        trainable_pct=p["trainable_pct"],
        n_train=len(train_ds),
        n_val=len(val_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, bundle.processor),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate(b, bundle.processor),
    )

    trainable = [pp for pp in bundle.model.parameters() if pp.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    step = 0
    log_window: list[float] = []
    t0 = time.time()
    train_iter = iter(train_loader)
    while step < cfg["training"]["steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = bundle.model(**batch)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        loss_val = out.loss.item()
        result.losses.append(loss_val)
        log_window.append(loss_val)

        if step % cfg["training"]["log_every"] == 0:
            print(f"  step {step:>4}  train_loss={loss_val:.4f}")

        if step > 0 and step % cfg["training"]["eval_every"] == 0:
            val_loss = evaluate(bundle, val_loader, device)
            samples = sample_generations(
                bundle, val_ds, cfg["generation"]["n_samples"], cfg["generation"], device
            )
            ep = EvalPoint(
                step=step,
                train_loss=sum(log_window) / max(len(log_window), 1),
                val_loss=val_loss,
                samples=samples,
            )
            result.eval_points.append(ep)
            print(f"  [eval] step {step}: val_loss={val_loss:.4f}")
            for s in samples[:3]:
                print(f"    {s['filename']}: gen={s['generated']!r}")
            log_window.clear()

        step += 1

    val_loss = evaluate(bundle, val_loader, device)
    samples = sample_generations(
        bundle, val_ds, cfg["generation"]["n_samples"], cfg["generation"], device
    )
    result.eval_points.append(EvalPoint(
        step=step,
        train_loss=sum(log_window) / max(len(log_window), 1) if log_window else float("nan"),
        val_loss=val_loss,
        samples=samples,
    ))
    result.wall_seconds = time.time() - t0

    print()
    print("=" * 70)
    print(f"FINAL  val_loss={val_loss:.4f}  wall={result.wall_seconds:.1f}s")
    print("=" * 70)
    for s in samples:
        print(f"  {s['filename']}")
        print(f"    ref: {s['reference']}")
        print(f"    gen: {s['generated']}")

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(result, out_dir / "results.md", cfg)
    (out_dir / "samples.json").write_text(
        json.dumps([ep.samples for ep in result.eval_points], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote: {out_dir / 'results.md'}")
    print(f"       {out_dir / 'samples.json'}")

    if cfg["output"].get("save_adapter"):
        ckpt_dir = paths.EXP_CHECKPOINTS / cfg["output"]["results_subdir"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        bundle.model.save_pretrained(ckpt_dir)
        bundle.processor.tokenizer.save_pretrained(ckpt_dir)
        print(f"       {ckpt_dir} (adapter + tokenizer)")


if __name__ == "__main__":
    main()
