"""Multi-reference caption metrics for Bangla.

Implements the paper-plan §C3 metric battery on (hypothesis, references)
pairs:

  - BLEU-1..4 and CIDEr via pycocoevalcap (corpus-level, multi-reference —
    the standard scorers used by prior Bangla captioning work, so numbers
    are comparable to BAN-Cap / Bornon baselines). Only the Bleu/Cider
    scorers are used; the Java-dependent PTBTokenizer/METEOR/SPICE paths
    are avoided on purpose.
  - BERTScore with the multilingual backbone bert-score selects for
    ``lang="bn"`` (bert-base-multilingual-cased). Multi-reference: max
    over references, bert-score's built-in behaviour.
  - M-CLIPScore: CLIPScore (Hessel et al. 2021, w=2.5) computed with a
    multilingual CLIP text tower so Bangla text embeds into the same
    space as the image. Requires ``sentence-transformers``; the caller
    can skip this metric if the package is unavailable.

Tokenization: whitespace after separating the danda (।) and common
punctuation from adjacent words. No stemming, no lowercasing (Bangla has
no case). This matches the whitespace-token convention of prior Bangla
captioning papers; note it in the paper's evaluation section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PUNCT_RE = re.compile(r"([।॥!?,;:.])")
_WS_RE = re.compile(r"\s+")


def tokenize_bn(text: str) -> str:
    """Whitespace-tokenize Bangla text, splitting danda/punctuation off words.

    Returns a space-joined token string (the format pycocoevalcap expects).
    """
    text = _PUNCT_RE.sub(r" \1 ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass
class NGramScores:
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    cider: float


def ngram_scores(hyps: list[str], refs: list[list[str]]) -> NGramScores:
    """Corpus-level BLEU-1..4 + CIDEr over multi-reference pairs.

    Parameters
    ----------
    hyps : one hypothesis string per image.
    refs : list of reference lists, aligned with ``hyps`` (BAN-Cap: 5 each).
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [tokenize_bn(r) for r in rr] for i, rr in enumerate(refs)}
    res = {i: [tokenize_bn(h)] for i, h in enumerate(hyps)}

    bleu, _ = Bleu(4).compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return NGramScores(
        bleu1=float(bleu[0]),
        bleu2=float(bleu[1]),
        bleu3=float(bleu[2]),
        bleu4=float(bleu[3]),
        cider=float(cider),
    )


def bertscore_bn(
    hyps: list[str],
    refs: list[list[str]],
    *,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, float]:
    """Mean multi-reference BERTScore P/R/F1 for Bangla (lang='bn')."""
    from bert_score import score as _bs

    P, R, F1 = _bs(
        hyps,
        refs,
        lang="bn",
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def m_clipscore(
    image_paths: list[Path],
    hyps: list[str],
    *,
    text_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    image_model_name: str = "clip-ViT-B-32",
    device: str = "cuda",
    batch_size: int = 32,
    w: float = 2.5,
) -> float:
    """Reference-free M-CLIPScore: mean of ``w * max(cos(img, text), 0)``.

    Uses sentence-transformers' multilingual CLIP text tower (trained to
    align with the ViT-B/32 image space) so Bangla hypotheses score
    without translation. Raises ImportError if sentence-transformers is
    not installed — callers should treat the metric as optional.
    """
    import numpy as np
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    img_model = SentenceTransformer(image_model_name, device=device)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    img_emb = img_model.encode(
        images, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    del img_model, images

    txt_model = SentenceTransformer(text_model_name, device=device)
    txt_emb = txt_model.encode(
        hyps, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    del txt_model

    cos = (img_emb * txt_emb).sum(axis=1)
    return float(np.mean(w * np.clip(cos, 0.0, None)))
