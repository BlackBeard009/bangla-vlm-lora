"""CLI entry point for the tokenizer fertility audit.

Usage:
    python scripts/tokenizer_audit.py --config configs/tokenizer_audit.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

# Make `src` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer.fertility import audit_corpus  # noqa: E402
import paths  # noqa: E402


import re as _re
_SENT_SPLIT = _re.compile(r"(?<=[।!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def load_corpus(cfg: dict) -> list[str]:
    corpus_cfg = cfg["corpus"]
    if "hf_dataset" in corpus_cfg:
        from datasets import load_dataset
        spec = corpus_cfg["hf_dataset"]
        kwargs = {"split": spec["split"]}
        if spec.get("config"):
            kwargs["name"] = spec["config"]
        ds = load_dataset(spec["name"], **kwargs, streaming=True)

        texts: list[str] = []
        target = spec.get("max_samples") or 10_000
        split_sents = spec.get("split_into_sentences", False)
        for row in ds:
            raw = str(row[spec["text_field"]])
            if split_sents:
                for s in _split_sentences(raw):
                    if len(s) < 20:
                        continue
                    texts.append(s)
                    if len(texts) >= target:
                        break
            else:
                texts.append(raw)
            if len(texts) >= target:
                break
        return texts[:target]
    if "local_file" in corpus_cfg:
        with open(corpus_cfg["local_file"], encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("corpus config must specify hf_dataset or local_file")


def write_csv(results, out_path: Path) -> None:
    fieldnames = list(results[0].as_dict().keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.as_dict())


def write_markdown(results, out_path: Path, corpus_size: int) -> None:
    cols = [
        ("name", "Tokenizer"),
        ("role", "Role"),
        ("vocab_size", "Vocab"),
        ("bn_vocab_tokens", "BN vocab"),
        ("fertility", "Fertility ↓"),
        ("unk_rate", "UNK% ↓"),
        ("roundtrip_preserved", "Round-trip ↑"),
    ]
    lines = [
        "# Tokenizer fertility audit",
        "",
        f"Corpus: **{corpus_size}** Bangla sentences.",
        "",
        "| " + " | ".join(label for _, label in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in results:
        d = r.as_dict()
        if r.failed:
            row = [d["name"], f"FAILED: {r.error}", "-", "-", "-", "-", "-"]
        else:
            row = [
                d["name"],
                d["role"],
                str(d["vocab_size"]),
                str(d["bn_vocab_tokens"]),
                f"{d['fertility']:.3f}",
                f"{d['unk_rate']:.2%}",
                f"{d['roundtrip_preserved']:.2%}",
            ]
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading corpus ...")
    texts = load_corpus(cfg)
    print(f"  {len(texts)} sentences loaded")

    results = audit_corpus(cfg["tokenizers"], texts)

    paths.ensure_dirs()
    out_dir = paths.EXP_RESULTS / "tokenizer_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / cfg["output"]["results_csv"]
    md_path = out_dir / cfg["output"]["results_md"]
    write_csv(results, csv_path)
    write_markdown(results, md_path, corpus_size=len(texts))

    print(f"\nWrote:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
