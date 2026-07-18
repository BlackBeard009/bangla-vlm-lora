"""Reproduce the Bornon transformer captioner on BanglaLekha.

Metric-validity study (docs/metric_validity/bornon_reproduction.md):
train the published recipe (arXiv:2109.05218, Table 2 best BanglaLekha
row: 3 layers / 1 head, corpus BLEU-4 0.408) from the paper's stated
specification, then score the SAME trained model three ways:

  1. "their protocol": NLTK corpus BLEU on Keras-tokenized text where
     BOTH hypotheses and references pass through the top-5,000-word
     tokenizer (rare words silently dropped).
  2. "plain": NLTK corpus BLEU on whitespace tokens, no truncation.
  3. Our honest battery (pycocoevalcap BLEU/CIDEr + BERTScore-bn +
     M-CLIPScore) via src.evaluation.caption_metrics.

The delta between (1)/(2)/(3) quantifies how much of a published-style
number is protocol rather than model.

Usage:
    python scripts/bornon_repro.py --config configs/bornon_repro.yaml
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
from src.models.bornon_transformer import (  # noqa: E402
    BornonTransformer,
    VaswaniLR,
)
from src.tokenizer.keras_style import KerasStyleTokenizer  # noqa: E402

START, END = "startseq", "endseq"


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_entries(captions_path: Path, images_dir: Path) -> list[dict]:
    with captions_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    return [
        e for e in entries
        if e.get("caption") and (images_dir / e["filename"]).exists()
    ]


def split_bornon(entries: list[dict], seed: int) -> tuple[list, list, list]:
    """Paper Table 1: BanglaLekha 7154 train / 1000 val / 1000 test."""
    import random

    idx = list(range(len(entries)))
    random.Random(seed).shuffle(idx)
    test = [entries[i] for i in idx[:1000]]
    val = [entries[i] for i in idx[1000:2000]]
    train = [entries[i] for i in idx[2000:]]
    return train, val, test


def extract_features(
    entries: list[dict], images_dir: Path, cache_path: Path, device: str
) -> torch.Tensor:
    """InceptionV3 (softmax removed) 299x299 -> (N, 64, 2048) float16."""
    if cache_path.exists():
        feats = torch.load(cache_path, map_location="cpu")
        if feats.shape[0] == len(entries):
            print(f"[features] cache hit: {cache_path} {tuple(feats.shape)}")
            return feats
        print("[features] cache size mismatch — re-extracting")

    from PIL import Image
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.transforms import functional as TF

    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    net.eval().to(device)
    # Forward through everything up to (not including) avgpool — the
    # Keras include_top=False equivalent: output (N, 2048, 8, 8).
    layers = [
        net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
        net.maxpool1, net.Conv2d_3b_1x1, net.Conv2d_4a_3x3, net.maxpool2,
        net.Mixed_5b, net.Mixed_5c, net.Mixed_5d, net.Mixed_6a,
        net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
        net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
    ]
    backbone = torch.nn.Sequential(*layers)

    feats = torch.empty(len(entries), 64, 2048, dtype=torch.float16)
    batch, keep = [], []
    t0 = time.time()

    def flush() -> None:
        if not batch:
            return
        with torch.no_grad():
            x = torch.stack(batch).to(device)
            y = backbone(x)  # (B, 2048, 8, 8)
            y = y.flatten(2).transpose(1, 2)  # (B, 64, 2048)
        for j, row in zip(keep, y.to(torch.float16).cpu()):
            feats[j] = row
        batch.clear()
        keep.clear()

    for i, e in enumerate(entries):
        with Image.open(images_dir / e["filename"]) as im:
            img = im.convert("RGB")
        x = TF.to_tensor(TF.resize(img, [299, 299]))
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        batch.append(x)
        keep.append(i)
        if len(batch) == 64:
            flush()
            if (i + 1) % 1024 == 0:
                print(f"  [features] {i + 1}/{len(entries)} "
                      f"({time.time() - t0:.0f}s)")
    flush()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, cache_path)
    print(f"[features] wrote {cache_path} ({time.time() - t0:.0f}s)")
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg = cfg["data"]
    captions_path = resolve_path(dcfg["captions_path"])
    images_dir = resolve_path(dcfg["images_dir"])
    entries = load_entries(captions_path, images_dir)
    train_e, val_e, test_e = split_bornon(entries, cfg["seed"])
    print(f"[data] train={len(train_e)} val={len(val_e)} test={len(test_e)}")

    # ---- tokenizer: Keras-style, top-5000, fit on TRAIN captions ----
    tcfg = cfg["tokenizer"]
    train_caps = [f"{START} {c.strip()} {END}"
                  for e in train_e for c in e["caption"]]
    tok = KerasStyleTokenizer(num_words=tcfg["num_words"])
    tok.fit_on_texts(train_caps)
    vocab_size = min(tcfg["num_words"], len(tok.word_index) + 1)
    start_id = tok.word_index[START]
    end_id = tok.word_index[END]
    train_seqs = tok.texts_to_sequences(train_caps)
    max_len = max(len(s) for s in train_seqs)
    print(f"[tok] fitted={len(tok.word_index)} kept={vocab_size - 1} "
          f"max_len={max_len} start={start_id} end={end_id}")

    # ---- features (all entries at once, one cache) ----
    all_entries = train_e + val_e + test_e
    feats = extract_features(
        all_entries, images_dir,
        paths.DATA_PROCESSED / "bornon_repro" / "inception_features.pt",
        device,
    )
    offsets = {"train": 0, "val": len(train_e), "test": len(train_e) + len(val_e)}

    # ---- model ----
    mcfg = cfg["model"]
    model = BornonTransformer(
        vocab_size=vocab_size,
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_model=mcfg["d_model"],
        dff=mcfg["dff"],
        dropout=mcfg["dropout"],
        max_len=max_len + 8,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params  layers={mcfg['num_layers']} "
          f"heads={mcfg['num_heads']} d_model={mcfg['d_model']}")

    # ---- training: 50 epochs, batch 64, Vaswani LR, CE-with-pad-mask ----
    samples = [
        (offsets["train"] + i, seq)
        for i, e in enumerate(train_e)
        for seq in tok.texts_to_sequences(
            [f"{START} {c.strip()} {END}" for c in e["caption"]]
        )
    ]
    print(f"[train] samples={len(samples)}")
    opt = torch.optim.Adam(model.parameters(), lr=0.0,
                           betas=(0.9, 0.98), eps=1e-9)
    sched = VaswaniLR(opt, mcfg["d_model"], warmup=cfg["training"]["warmup"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    bsz = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]

    def make_batch(chunk: list[tuple[int, list[int]]]):
        f = torch.stack([feats[i] for i, _ in chunk]).to(device, torch.float32)
        t = max(len(s) for _, s in chunk)
        toks = torch.zeros(len(chunk), t, dtype=torch.long)
        for r, (_, s) in enumerate(chunk):
            toks[r, : len(s)] = torch.tensor(s)
        return f, toks.to(device)

    t0 = time.time()
    order_rng = torch.Generator().manual_seed(cfg["seed"])
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(samples), generator=order_rng).tolist()
        total, n_batches = 0.0, 0
        for b0 in range(0, len(samples), bsz):
            chunk = [samples[i] for i in perm[b0 : b0 + bsz]]
            f, toks = make_batch(chunk)
            logits = model(f, toks[:, :-1])
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sched.step()
            opt.step()
            total += loss.item()
            n_batches += 1
        history.append(total / n_batches)
        print(f"  epoch {epoch + 1:>3}/{epochs}  loss={history[-1]:.4f}  "
              f"({time.time() - t0:.0f}s)")

    # ---- greedy decode the 1000-image test split ----
    model.eval()
    hyps: list[str] = []
    for b0 in range(0, len(test_e), bsz):
        chunk_idx = [offsets["test"] + i
                     for i in range(b0, min(b0 + bsz, len(test_e)))]
        f = torch.stack([feats[i] for i in chunk_idx]).to(device, torch.float32)
        ids = model.generate_greedy(
            f, start_id=start_id, end_id=end_id, max_new_tokens=max_len
        )
        hyps.extend(tok.sequences_to_texts(ids))
    refs = [[c.strip() for c in e["caption"]] for e in test_e]

    # ---- scoring, three ways ----
    from nltk.translate.bleu_score import corpus_bleu

    def nltk_bleu(hyp_tokens: list[list[str]],
                  ref_tokens: list[list[list[str]]]) -> dict[str, float]:
        weights = {
            "bleu1": (1.0,),
            "bleu2": (0.5, 0.5),
            "bleu3": (1 / 3, 1 / 3, 1 / 3),
            "bleu4": (0.25, 0.25, 0.25, 0.25),
        }
        return {k: float(corpus_bleu(ref_tokens, hyp_tokens, weights=w))
                for k, w in weights.items()}

    # (1) their protocol: hyps AND refs round-tripped through the
    #     top-5000 tokenizer (rare words silently dropped).
    hyp_trunc = [s.split() for s in tok.sequences_to_texts(
        tok.texts_to_sequences(hyps))]
    ref_trunc = [
        [t.split() for t in tok.sequences_to_texts(tok.texts_to_sequences(rr))]
        for rr in refs
    ]
    their = nltk_bleu(hyp_trunc, ref_trunc)

    # (2) plain: whitespace tokens, full vocabulary.
    plain = nltk_bleu([h.split() for h in hyps],
                      [[r.split() for r in rr] for rr in refs])

    # (3) honest battery (same scorers as every other row in the study).
    from src.evaluation.caption_metrics import ngram_scores

    battery = ngram_scores(hyps, refs).__dict__
    mcfg2 = cfg.get("metrics", {})
    if mcfg2.get("bertscore"):
        from src.evaluation.caption_metrics import bertscore_bn

        battery.update(bertscore_bn(
            hyps, refs, batch_size=mcfg2.get("bertscore_batch_size", 64)))

    results = {
        "their_protocol_top5000_nltk": their,
        "plain_whitespace_nltk": plain,
        "honest_battery_pycoco": battery,
        "n_test": len(test_e),
        "mean_len_words": sum(len(h.split()) for h in hyps) / len(hyps),
        "n_empty": sum(not h.strip() for h in hyps),
        "unique_output_tokens": len({w for h in hyps for w in h.split()}),
        "train_loss_final": history[-1],
        "wall_seconds": time.time() - t0,
    }

    paths.ensure_dirs()
    out = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "predictions.json").write_text(
        json.dumps(
            {"filenames": [e["filename"] for e in test_e],
             "references": refs, "predictions": hyps},
            ensure_ascii=False, indent=2),
        encoding="utf-8")
    if cfg["output"].get("save_model"):
        ckpt = paths.EXP_CHECKPOINTS / cfg["output"]["results_subdir"]
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / "model.pt")
        (ckpt / "word_index.json").write_text(
            json.dumps(tok.word_index, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    for k in ("their_protocol_top5000_nltk", "plain_whitespace_nltk"):
        print(f"{k}: " + "  ".join(f"{m}={v:.3f}" for m, v in results[k].items()))
    print(f"honest pycoco: " + "  ".join(
        f"{m}={v:.3f}" for m, v in battery.items() if isinstance(v, float)))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
