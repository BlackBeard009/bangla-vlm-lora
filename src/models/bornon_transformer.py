"""Faithful reimplementation of the Bornon transformer captioner.

Bornon (Shah et al., arXiv:2109.05218) — the strongest published
BanglaLekha number (corpus BLEU-4 0.408, Table 2, 3 layers / 1 head).
No code was released; this module reimplements the architecture from
the paper's stated specification for the metric-validity reproduction
study (docs/metric_validity/bornon_reproduction.md).

Stated in the paper (Sections 5, 6, 10):
  - InceptionV3, softmax head removed, 299x299 input, 8x8x2048 feature
    map -> 64 tokens of 2048 dims, precomputed and cached.
  - Standard Vaswani encoder-decoder transformer: image tokens into the
    encoder, tokenized caption into the decoder; multi-head attention
    with padding masks, look-ahead mask on the decoder.
  - Keras text tokenizer, top-5,000-word vocabulary.
  - 50 epochs, batch 64, Adam with the Vaswani LR schedule
    (warmup_steps=4000), SparseCategoricalCrossentropy.

NOT stated (assumptions, chosen to match the TensorFlow tutorial
lineage the paper's wording follows; flagged in the study doc):
  - d_model=512, dff=2048, dropout=0.1 (Vaswani defaults).
  - Sinusoidal positional encoding on decoder tokens; a learned linear
    projection (2048 -> d_model) on encoder tokens, no positional
    encoding on the 64 image positions.
  - Greedy decoding (the paper discusses beam search only for prior
    work's models).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class BornonTransformer(nn.Module):
    """Encoder-decoder transformer over precomputed InceptionV3 tokens."""

    def __init__(
        self,
        vocab_size: int,
        *,
        num_layers: int = 3,
        num_heads: int = 1,
        d_model: int = 512,
        dff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 64,
        feature_dim: int = 2048,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.register_buffer(
            "pos_encoding", sinusoidal_positions(max_len, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # post-LN, as in Vaswani / Keras tutorials
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self, features: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """features: (B, 64, 2048); tokens: (B, T) — returns (B, T, V)."""
        src = self.feature_proj(features)
        tgt = self.embedding(tokens) * math.sqrt(self.d_model)
        tgt = self.dropout(tgt + self.pos_encoding[: tokens.shape[1]])
        causal = nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1], device=tokens.device
        )
        out = self.transformer(
            src,
            tgt,
            tgt_mask=causal,
            tgt_key_padding_mask=(tokens == self.pad_id),
        )
        return self.lm_head(out)

    @torch.no_grad()
    def generate_greedy(
        self,
        features: torch.Tensor,
        *,
        start_id: int,
        end_id: int,
        max_new_tokens: int,
    ) -> list[list[int]]:
        """Greedy decode a batch of feature maps; returns token id lists
        (start/end stripped, stops per-sequence at end_id)."""
        self.eval()
        bsz = features.shape[0]
        device = features.device
        tokens = torch.full((bsz, 1), start_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits = self.forward(features, tokens)
            nxt = logits[:, -1].argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, end_id), nxt)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            finished |= nxt == end_id
            if bool(finished.all()):
                break
        out: list[list[int]] = []
        for row in tokens[:, 1:].tolist():
            ids = []
            for t in row:
                if t == end_id:
                    break
                ids.append(t)
            out.append(ids)
        return out


class VaswaniLR:
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5) — Eq. 6."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, d_model: int, warmup: int = 4000
    ) -> None:
        self.optimizer = optimizer
        self.scale = d_model ** -0.5
        self.warmup = warmup
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.scale * min(
            self.step_num ** -0.5, self.step_num * self.warmup ** -1.5
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
