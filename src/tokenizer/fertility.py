"""Tokenizer fertility audit for Bangla text.

For each candidate tokenizer, compute:
- fertility = subtokens / whitespace-words
- unk_rate  = UNK tokens / total tokens
- roundtrip = 1 - char_error_rate of decode(encode(x)) vs x
- bn_vocab  = # vocab entries that contain at least one Bangla codepoint
- vocab_size

`audit_tokenizer` and `audit_corpus` are the two public entry points.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from transformers import AutoTokenizer

# Bangla Unicode block: U+0980..U+09FF
_BN_RE = re.compile(r"[ঀ-৿]")


def _has_bangla(s: str) -> bool:
    return bool(_BN_RE.search(s))


def _whitespace_word_count(texts: Iterable[str]) -> int:
    return sum(len(t.split()) for t in texts)


def _char_error_rate(ref: str, hyp: str) -> float:
    """Simple normalized Levenshtein at the character level.

    Returns 0.0 for a perfect match, 1.0 for total disagreement.
    Uses an O(len(ref) * len(hyp)) DP; fine for caption-length strings.
    """
    if ref == hyp:
        return 0.0
    if not ref:
        return 1.0
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * n
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (rc != hc),
            )
        prev = cur
    return prev[n] / max(m, 1)


@dataclass
class TokenizerStats:
    name: str
    role: str
    vocab_size: int
    bn_vocab_tokens: int
    n_input_words: int
    n_subtokens: int
    n_unk: int
    fertility: float
    unk_rate: float
    roundtrip_preserved: float  # 1 - mean CER
    failed: bool = False
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _count_bangla_vocab(tok) -> int:
    try:
        vocab = tok.get_vocab()
    except Exception:
        return -1
    return sum(1 for piece in vocab.keys() if _has_bangla(piece))


def audit_tokenizer(
    name: str,
    role: str,
    texts: list[str],
    *,
    trust_remote_code: bool = False,
) -> TokenizerStats:
    """Run the audit for a single tokenizer.

    Errors are caught so one bad tokenizer doesn't halt the suite.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            name, trust_remote_code=trust_remote_code, use_fast=True
        )
    except Exception as e:
        return TokenizerStats(
            name=name, role=role,
            vocab_size=0, bn_vocab_tokens=0,
            n_input_words=0, n_subtokens=0, n_unk=0,
            fertility=float("nan"), unk_rate=float("nan"),
            roundtrip_preserved=float("nan"),
            failed=True, error=f"load_failed: {type(e).__name__}: {e}",
        )

    unk_id = tok.unk_token_id  # may be None for byte-level BPE
    n_words = _whitespace_word_count(texts)
    n_sub = 0
    n_unk = 0
    cer_sum = 0.0
    cer_n = 0

    for text in texts:
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            continue
        n_sub += len(ids)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
        try:
            decoded = tok.decode(ids, skip_special_tokens=True)
        except Exception:
            decoded = ""
        cer_sum += _char_error_rate(text, decoded)
        cer_n += 1

    fertility = n_sub / max(n_words, 1)
    unk_rate = (n_unk / n_sub) if n_sub else 0.0
    roundtrip = 1.0 - (cer_sum / max(cer_n, 1))

    return TokenizerStats(
        name=name,
        role=role,
        vocab_size=len(tok),
        bn_vocab_tokens=_count_bangla_vocab(tok),
        n_input_words=n_words,
        n_subtokens=n_sub,
        n_unk=n_unk,
        fertility=fertility,
        unk_rate=unk_rate,
        roundtrip_preserved=roundtrip,
    )


def audit_corpus(
    tokenizer_specs: list[dict],
    texts: list[str],
) -> list[TokenizerStats]:
    """Audit every tokenizer in `tokenizer_specs` against `texts`."""
    results = []
    for spec in tokenizer_specs:
        print(f"[audit] {spec['name']} ...", flush=True)
        result = audit_tokenizer(
            name=spec["name"],
            role=spec.get("role", ""),
            texts=texts,
            trust_remote_code=spec.get("trust_remote_code", False),
        )
        if result.failed:
            print(f"  FAILED: {result.error}", flush=True)
        else:
            print(
                f"  fertility={result.fertility:.3f}  "
                f"unk_rate={result.unk_rate:.3%}  "
                f"roundtrip={result.roundtrip_preserved:.3%}  "
                f"bn_vocab={result.bn_vocab_tokens}",
                flush=True,
            )
        results.append(result)
    return results
