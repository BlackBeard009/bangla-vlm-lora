"""Margin-normalized semantic similarity (SemMargin) for caption evaluation.

Motivation: plain embedding similarity between a
hypothesis and the references — BERTScore, SBERT cosine, M-CLIPScore —
is gamed by generic captions, which sit near the semantic centroid of
the whole corpus (a constant no-vision 3-word caption reaches
BERTScore-F1 0.798 on BanglaLekha). The fix is contrastive: score
similarity to THIS image's references minus similarity to OTHER
images' references, so corpus-prior similarity subtracts out.

    raw(h_i)    = mean_r cos(e(h_i), e(ref_{i,r}))
    prior(h_i)  = mean_{j != i} mean_r cos(e(h_i), e(ref_{j,r}))
    margin(h_i) = raw(h_i) - prior(h_i)

Also reported:
  - rank variant: the percentile of raw(h_i) within the distribution
    of {mean-ref-similarity of h_i to every image j} — 1.0 means the
    hypothesis is closer to its own image's references than to any
    other image's (reference-side self-retrieval).
  - a human ceiling: score each reference held out against the
    remaining references of its image, same formulas.

Encoder: any sentence-transformers model; default LaBSE
(language-agnostic BERT sentence embeddings, strong Bangla coverage).
Corpus-level score = mean over images. All pairwise similarities are
computed in one matrix product, so the full 809-image battery is
seconds of GPU time after encoding.
"""

from __future__ import annotations

import numpy as np


_MODEL_CACHE: dict[str, object] = {}


def _encode(texts: list[str], model_name: str, batch_size: int = 128):
    # Import pyarrow before sentence_transformers: on this workstation
    # (torch 2.11 + pyarrow 24.0, Windows), letting sentence_transformers'
    # own import chain load pyarrow.lib mid-way dies with an access
    # violation (exit 5, no traceback). Preloading is a reliable fix.
    # The model is cached per process — reconstructing it per call both
    # wastes time and re-rolls the same DLL fault.
    import pyarrow  # noqa: F401
    import pyarrow.dataset  # noqa: F401
    from sentence_transformers import SentenceTransformer

    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model = _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def semantic_margin(
    hyps: list[str],
    refs: list[list[str]],
    *,
    model_name: str = "sentence-transformers/LaBSE",
    batch_size: int = 128,
) -> dict[str, float]:
    """Corpus-level SemMargin scores for aligned (hyps, refs).

    Returns dict with raw / prior / margin / rank means.
    """
    n = len(hyps)
    flat_refs = [r for rr in refs for r in rr]
    ref_counts = [len(rr) for rr in refs]

    hyp_emb = _encode(hyps, model_name, batch_size)
    ref_emb = _encode(flat_refs, model_name, batch_size)

    # (n_hyp, n_all_refs) cosine matrix (embeddings are L2-normalized).
    sims = hyp_emb @ ref_emb.T

    # Mean similarity of every hypothesis to every image's reference set:
    # (n_hyp, n_images) — column j is mean over image j's refs.
    per_image = np.empty((n, n), dtype=np.float64)
    start = 0
    for j, c in enumerate(ref_counts):
        per_image[:, j] = sims[:, start : start + c].mean(axis=1)
        start += c

    own = per_image[np.arange(n), np.arange(n)]
    other_sum = per_image.sum(axis=1) - own
    prior = other_sum / (n - 1)
    margin = own - prior
    # Percentile rank of the own-image similarity among all images.
    rank = (per_image < own[:, None]).sum(axis=1) / (n - 1)

    return {
        "sem_raw": float(own.mean()),
        "sem_prior": float(prior.mean()),
        "sem_margin": float(margin.mean()),
        "sem_rank": float(rank.mean()),
    }


def human_ceiling(
    refs: list[list[str]],
    *,
    model_name: str = "sentence-transformers/LaBSE",
    batch_size: int = 128,
) -> dict[str, float]:
    """SemMargin of held-out references: ref[0] scored against the
    remaining references of each image. Requires >= 2 refs per image."""
    usable = [rr for rr in refs if len(rr) >= 2]
    hyps = [rr[0] for rr in usable]
    rest = [rr[1:] for rr in usable]
    return semantic_margin(
        hyps, rest, model_name=model_name, batch_size=batch_size
    )
