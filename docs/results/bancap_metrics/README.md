# First real metrics — bancap_full (5K) vs bancap_full_10k adapters

**Date:** 2026-07-05 · **Hardware:** RTX 4060 Ti 8 GB (local; first
post-Colab results) · **Script:** `scripts/score_captions.py` with
`configs/score_captions_bancap{,_10k}.yaml`.

Full BAN-Cap val split (809 images × 5 native references), three decode
variants per adapter. Scorers: pycocoevalcap BLEU-1..4 + CIDEr
(whitespace+danda tokenization), BERTScore lang=bn
(bert-base-multilingual-cased), M-CLIPScore (w=2.5,
clip-ViT-B-32-multilingual-v1 text tower). Raw dumps:
`paths.EXP_RESULTS/score_captions_bancap{,_10k}/`.

## Headline table (best decode variant: `min8_no_early_stop`)

| Adapter | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | BERTScore-F1 | M-CLIPScore |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bancap_full` (5K)      | 0.437 | 0.130 | 0.031 | 0.009 | 0.153 | 0.747 | 0.552 |
| `bancap_full_10k` (10K) | 0.486 | 0.133 | 0.023 | 0.000 | **0.179** | 0.753 | 0.547 |
| `bancap_img_lora_10k` (10K + image-tower) | **0.498** | **0.145** | **0.032** | 0.000 | 0.172 | **0.758** | 0.546 |

### Image-tower LoRA run (2026-07-05, CLAUDE.md item 8)

`configs/bancap_img_lora_10k.yaml` — identical to `bancap_full_10k`
except: LoRA also on CLIP-ViT `q_proj/k_proj/v_proj/out_proj` (12
layers; text-only recipe silently missed them) + fully unfrozen
`visual_projection.visual_projection.0`. Trainable 110.7M (33.34%).
Wall 1,617 s (+14% vs text-only — ViT backprop is cheap).

- **Val CE flat** (4.489 vs 4.474) but **precision metrics up across
  the board**: BLEU-1/2 up, BLEU-3 recovers the 10K-text regression
  (0.023 → 0.032), BERTScore-F1 best of all runs. CE and generation
  quality dissociate — worth a sentence in the paper.
- **Decode degeneration halves**: 7/809 empty at unconstrained beam-4
  vs 26/809 for text-only 10K.
- CIDEr slightly down (0.179 → 0.172), M-CLIPScore flat. Gains are
  real but modest — consistent with the diagnosis that the 6-layer
  decoder, not the image pathway, is the binding constraint.
- BLEU-4 still 0.000 at 10K steps regardless of placement. Composition
  needs a bigger decoder (Qwen2-VL-2B next), not more LoRA surface.

All variants, both adapters: see `metrics.json` in the results dirs.
Training: 10K val CE 5.80 → **4.474** (monotonic, no overfitting; step-5000
value 4.642 reproduces the Colab T4 run's 4.64 across platform +
transformers 4.x→5.x). Wall: 1,414 s for 10K steps.

## What doubling the training budget bought — and didn't

- **Content selection improved:** BLEU-1 +0.049, CIDEr +0.026,
  BERTScore-F1 +0.006. More/better content words per caption.
- **Composition did NOT improve:** BLEU-3 slightly down, BLEU-4 → 0.000.
  Longer n-gram structure is flat at both budgets. This is the metric
  confirmation of the sentence-diagnostics conclusion: the 6-layer
  GiT-base decoder is composition-bound, and more steps cannot buy
  grammar.
- **Unconstrained decode degenerated with more training:**
  `baseline_beam4` on the 10K adapter produces `একটি একটি …` loops and
  **26/809 empty captions** (5K: 0). Lower CE sharpens the repetition
  attractor. With `no_repeat_ngram_size=3` the loops vanish (n_empty=0)
  and the 10K adapter is strictly better on unigram/semantic metrics —
  so this is a decode artifact, consistent with the decode-ablation
  finding, but it means **`min8_no_early_stop` (beam=4, min_new_tokens=8,
  no_repeat_ngram_size=3) must be the default decode from now on.**
- **Decode ranking is stable across adapters:** `min8_no_early_stop` >
  `baseline_beam4` ≈ `beam4_penalties` on nearly every metric, both
  adapters.

## Honest context vs. published baselines

BAN-Cap's own models (LREC 2022) report BLEU-4 ≈ 0.17–0.20; we are at
≈ 0.0. The current adapters are motivation-chain artifacts (vocabulary
bridge works; content words emerge) — not yet competitive systems. The
gap is the empirical argument for the next experiments in priority
order: image-tower LoRA (q_proj/k_proj/v_proj + out_proj +
visual_projection) and the bigger-decoder model (Qwen2-VL-2B locally;
PaliGemma-3B needs >8 GB → Colab).

Semantic metrics (BERTScore ≈ 0.75, M-CLIPScore ≈ 0.55) are far more
forgiving than n-gram metrics on the same outputs — first data point for
the paper's §C3 metric-disagreement story.
