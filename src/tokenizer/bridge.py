"""Vocabulary + embedding bridge for adapting an English VLM tokenizer to Bangla.

Extends an existing tokenizer's vocabulary with Bangla wordpieces drawn from a
Bangla-native BERT-family tokenizer (default BanglaBERT). After extension,
the model's input embeddings and LM head are resized; the new rows are
initialized according to one of three strategies:

    random : leave PyTorch's default init (small normal noise)
    mean   : set every new row to the mean of the original rows
    donor  : copy from the donor model's embedding (where token matches)

The bridge is independent of LoRA wiring — apply LoRA after bridging.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


# Bangla Unicode block: U+0980..U+09FF
_BN_RE = re.compile(r"[ঀ-৿]")


def _has_bangla(s: str) -> bool:
    return bool(_BN_RE.search(s))


@dataclass
class BridgeReport:
    donor: str
    init_strategy: str
    n_existing: int          # vocab size before extension
    n_donor_total: int       # donor vocab size
    n_donor_with_bangla: int # tokens in donor vocab containing Bangla codepoints
    n_already_in_target: int # of those, how many already in target vocab
    n_added: int             # net new tokens added to target
    n_final: int             # vocab size after extension


def select_bangla_tokens_from_donor(
    donor_tokenizer_name: str,
    existing_vocab: set[str],
) -> tuple[list[str], BridgeReport]:
    """Return (tokens to add, partial BridgeReport for accounting)."""
    donor = AutoTokenizer.from_pretrained(donor_tokenizer_name)
    donor_vocab = donor.get_vocab()
    bangla_in_donor = [p for p in donor_vocab if _has_bangla(p)]
    new = sorted(p for p in bangla_in_donor if p not in existing_vocab)
    report = BridgeReport(
        donor=donor_tokenizer_name,
        init_strategy="(unknown)",
        n_existing=len(existing_vocab),
        n_donor_total=len(donor_vocab),
        n_donor_with_bangla=len(bangla_in_donor),
        n_already_in_target=len(bangla_in_donor) - len(new),
        n_added=len(new),
        n_final=len(existing_vocab) + len(new),
    )
    return new, report


@torch.no_grad()
def init_new_rows(
    model: torch.nn.Module,
    new_tokens: list[str],
    new_ids: list[int],
    strategy: str,
    *,
    donor_model_name: str | None = None,
    donor_tokenizer_name: str | None = None,
) -> None:
    """Initialize the new rows of input embeddings + output LM head in-place."""
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if in_emb is None or out_emb is None:
        raise RuntimeError("model lacks input or output embeddings; cannot bridge")

    n_new = len(new_ids)
    if n_new == 0:
        return

    if strategy == "random":
        return  # PyTorch's resize_token_embeddings already initialized randomly

    # Mean of the original (pre-extension) rows — exclude the new tail.
    orig_in = in_emb.weight[:-n_new]
    orig_out = out_emb.weight[:-n_new]
    mean_in = orig_in.mean(dim=0)
    mean_out = orig_out.mean(dim=0)

    if strategy == "mean":
        for tid in new_ids:
            in_emb.weight[tid] = mean_in
            out_emb.weight[tid] = mean_out
        return

    if strategy == "donor":
        if donor_model_name is None or donor_tokenizer_name is None:
            raise ValueError("donor init requires donor_model_name and donor_tokenizer_name")
        donor_model = AutoModel.from_pretrained(donor_model_name)
        donor_tok = AutoTokenizer.from_pretrained(donor_tokenizer_name)
        donor_in = donor_model.get_input_embeddings().weight  # (V_donor, D_donor)
        if donor_in.shape[1] != in_emb.weight.shape[1]:
            raise ValueError(
                f"donor hidden dim {donor_in.shape[1]} != target {in_emb.weight.shape[1]}"
            )
        donor_vocab = donor_tok.get_vocab()
        n_donor_hits = 0
        for token, tid in zip(new_tokens, new_ids):
            d_id = donor_vocab.get(token)
            if d_id is not None:
                in_emb.weight[tid] = donor_in[d_id]
                n_donor_hits += 1
            else:
                in_emb.weight[tid] = mean_in
            # LM head: donor's head shape almost certainly differs, fall back to mean
            out_emb.weight[tid] = mean_out
        # Tiny sanity print for the operator.
        print(
            f"  [bridge] donor init: {n_donor_hits}/{n_new} new tokens "
            f"received donor input-embedding rows; LM head rows = mean."
        )
        return

    raise ValueError(f"unknown init strategy: {strategy!r}")


def _build_extended_tokenizer(base_tokenizer, new_tokens: list[str]):
    """Build a fresh tokenizer with vocab = base + new_tokens.

    We can't simply call `base_tokenizer.add_tokens(...)`: in modern
    transformers all BERT-family tokenizers wrap the Rust `tokenizers`
    library, and adding many script-foreign tokens via the AddedVocabulary
    path panics with "bad split". So instead we save the base tokenizer,
    rewrite vocab.txt with the appended tokens, delete the cached fast
    state, and reload — producing a tokenizer where the new tokens are
    part of the *base* vocabulary, not "added" tokens.
    """
    save_dir = Path(tempfile.mkdtemp(prefix="git_bridged_tok_"))
    base_tokenizer.save_pretrained(save_dir)

    # Build the new vocab.txt: original tokens at their original IDs,
    # new tokens appended at the tail. (Order of new_tokens is fixed by caller.)
    base_vocab = base_tokenizer.get_vocab()  # {token: id}
    max_id = max(base_vocab.values())
    vocab_list = [None] * (max_id + 1)
    for tok, idx in base_vocab.items():
        vocab_list[idx] = tok
    # Sanity: BERT vocabs should be dense from 0..max_id with no holes
    if any(v is None for v in vocab_list):
        missing = [i for i, v in enumerate(vocab_list) if v is None]
        raise RuntimeError(
            f"base vocab has gaps at IDs {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    vocab_list.extend(new_tokens)

    (save_dir / "vocab.txt").write_text(
        "\n".join(vocab_list) + "\n", encoding="utf-8"
    )

    # Remove cached fast-tokenizer state so reload rebuilds from vocab.txt.
    for cached in ["tokenizer.json", "added_tokens.json"]:
        p = save_dir / cached
        if p.exists():
            p.unlink()

    return AutoTokenizer.from_pretrained(save_dir)


def bridge_vocabulary(
    bundle,
    *,
    donor_tokenizer: str = "csebuetnlp/banglabert",
    init_strategy: str = "mean",
    donor_model_for_init: str | None = "csebuetnlp/banglabert",
) -> BridgeReport:
    """Extend `bundle.processor.tokenizer` with Bangla tokens and resize embeddings.

    Mutates `bundle` in place (replaces the processor's tokenizer; resizes
    model embeddings; initializes the new rows). Returns a `BridgeReport`.
    """
    base_tokenizer = bundle.processor.tokenizer
    existing_vocab = set(base_tokenizer.get_vocab().keys())
    new_tokens, report = select_bangla_tokens_from_donor(donor_tokenizer, existing_vocab)
    report.init_strategy = init_strategy
    if not new_tokens:
        return report

    # Build the new tokenizer from a rewritten vocab.txt (see helper docstring).
    new_tokenizer = _build_extended_tokenizer(base_tokenizer, new_tokens)
    bundle.processor.tokenizer = new_tokenizer

    bundle.model.resize_token_embeddings(len(new_tokenizer))
    new_ids = new_tokenizer.convert_tokens_to_ids(new_tokens)

    # Detect silent drops, though _build_extended_tokenizer doesn't filter.
    if len(set(new_ids)) != len(new_tokens):
        report.n_added = len(set(new_ids))
        report.n_final = report.n_existing + report.n_added

    init_new_rows(
        bundle.model,
        new_tokens=new_tokens,
        new_ids=new_ids,
        strategy=init_strategy,
        donor_model_name=donor_model_for_init,
        donor_tokenizer_name=donor_tokenizer,
    )
    return report
