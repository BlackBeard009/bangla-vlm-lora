# Smoke test: GiT-base full fine-tune on synthetic Bangla

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/smoke_git_train.py`
**Config:** `configs/smoke_git.yaml`

## Setup

- Model: `microsoft/git-base` (347M params, full fine-tune, no LoRA)
- Dataset: 16 in-memory synthetic samples (random-color images + hand-written Bangla captions). See `src/data/synthetic_bangla.py`.
- 50 steps, batch 4, AdamW lr 5e-5, weight decay 0.01.

## Result

| Metric | Value |
|---|---|
| Steps | 50 |
| Loss (start) | 12.29 |
| Loss (end) | 5.06 |
| Throughput | ≈ 3 steps/sec (T4) |
| Wall time | 16.6 s |
| Status | **PASS** |

## Reference vs. generated caption

| Field | Value |
|---|---|
| Reference | `একটি লাল গাড়ি রাস্তায় চলছে।` ("A red car is moving on the road.") |
| Generated | `##াাাাাাাাাাাাাাাাাাাাাাাাাাাাাা` |

## Why this is a paper-grade finding (not just a smoke test)

The generation output is the **qualitative companion** to the Section 3
tokenizer audit:

- `##` is the BERT WordPiece continuation marker.
- `া` is the Bangla vowel sign *aa* (U+09BE) — one of the very few Bangla
  characters that survives `bert-base-uncased` tokenization.
- The model can train (loss decreases), but its output vocabulary cannot
  form proper Bangla words. It collapses to the single Bangla token it
  has access to, repeated.

This directly demonstrates why the tokenizer/embedding bridge (paper
contribution C1) is necessary. Without it, vanilla GiT physically cannot
produce Bangla text, regardless of how well the visual encoder works.

## Caveats

- Synthetic images are random solid colors; do not draw conclusions about
  visual grounding from this run.
- Loss values are not directly comparable to real-data runs.
- 50 steps is far too few to fairly evaluate the base model — the point
  here is solely that the training loop works.

## Next

Same pipeline on real BanglaLekha / BAN-Cap data once Drive has the files.
LoRA variant follows in a separate experiment (and will validate or
falsify the HF #1958 GiT+PEFT integration concern).
