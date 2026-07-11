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

## Qwen2-VL-2B QLoRA (2026-07-10) — vocabulary question answered

`scripts/qwen_qlora_train.py` + `configs/qwen_qlora_bancap.yaml`.
Attention-only LoRA (q/k/v/o_proj, LLM only, r=8, **0.178% trainable**,
2.18M params), 4-bit NF4 base, 3K steps × accum 4 = 12K samples,
2.7 h on the 4060 Ti. Deliberate mirror of the GiT V1 attention-only
ablation that failed with English output.

| System | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | BERTScore-F1 | M-CLIPScore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-VL-2B QLoRA | **0.532** | **0.331** | **0.177** | **0.082** | **0.300** | **0.804** | **0.561** |

- **809/809 outputs in Bangla** (zero-shot: 0/809). Bangla emerged in
  the 30-step smoke run already (~120 samples seen).
- **BLEU-4 wall broken**: 0.000 (all GiT runs) → 0.082. Compositional
  grammar comes from the 28-layer decoder, not from LoRA placement or
  training budget — exactly what the GiT diagnostics predicted.
- **Both directions of the vocabulary argument now closed
  experimentally:** GiT attention-only + missing vocab = no Bangla
  ever (V1); Qwen attention-only + covered vocab = fluent Bangla in
  120 samples. Vocabulary coverage is the deciding factor; the bridge
  is what buys it when the base model lacks it.
- BLEU-4 0.082 is within reach of BAN-Cap's published CNN/transformer
  baselines (~0.17–0.20) with 2.7 h of consumer-GPU training; rank /
  budget / decode ablations still open.
- Caveat for the paper: not an isolated single-variable comparison
  against GiT (base size, pretraining, and instruction tuning all
  change too). It answers the vocabulary question via the two
  attention-only endpoints, not "GiT vs Qwen".

## BanglaView curriculum stage 1 (2026-07-11) — data-ceiling test

`configs/qwen_qlora_banglaview.yaml`: same QLoRA recipe, corpus swapped
to BanglaView (31,783 Flickr30k images × 5 MT+post-edited Bangla
captions; 4× BAN-Cap scale). 6,000 steps × accum 4, 5.0 h, silver-val
CE 0.934 → 0.712. Scored on the SAME native BAN-Cap 809-val:

| Adapter | B1 | B4 | CIDEr | BERTScore-F1 | Unique output tokens |
|---|---:|---:|---:|---:|---:|
| BAN-Cap-only | 0.532 | 0.082 | 0.300 | 0.804 | 494 |
| BanglaView-only (zero BAN-Cap exposure) | 0.483 | 0.080 | 0.264 | 0.780 | **699** |

- **Output vocabulary +41% (494 → 699)** from 4× training data —
  direct confirmation that the token-collapse is data-limited (decode
  already ruled out via top-p). The scaling lever works.
- **Cross-corpus transfer is nearly free:** BanglaView-only matches
  in-domain BLEU-4 (0.080 vs 0.082) on a corpus it never saw;
  remaining gaps (CIDEr −0.036) are the expected domain shift from
  silver (translated) to gold (native) caption style.
- Stage 2 (resume this adapter, ~1.5K gold BAN-Cap steps) is queued —
  see `docs/SESSION_HANDOFF.md`. Hypothesis: diversity of stage 1 +
  style fit of gold data beats both single-corpus rows.

## Zero-shot Qwen2-VL-2B baseline (2026-07-10)

`scripts/zeroshot_vlm.py` + `configs/zeroshot_qwen2vl_bancap.yaml`.
Same 809 val images and references. Prompt ablation (7 variants) first:

- English instruction → fluent **English** captions; "Reply only in
  Bengali" ignored.
- Bangla instruction → grounding-coordinate junk (`dog(365,306),(879,903)`).
- Forced Bangla assistant prefix → Bangla script but hallucinated
  content (`ছবিতে ২০১২ সালের বার্ষিক প্রতিযোগিতায়…`).

Full-run row (best Bangla-eliciting prompt, outputs scored as-is;
**0/809 outputs contain any Bangla codepoint**):

| System | BLEU-1..4 | CIDEr | BERTScore-F1 | M-CLIPScore |
|---|---:|---:|---:|---:|
| Qwen2-VL-2B zero-shot | 0.000 all | 0.000 | 0.690 | **0.803** |

Two paper-grade findings:

1. **Tokenizer coverage ≠ generation capability.** Qwen2-VL's vocab
   handles Bangla (fertility audit) yet the 2B instruct model cannot
   produce it from image inputs. Sharpens the bridge story: our
   GiT+bridge (BLEU-1 0.498) beats a 6× larger multilingual VLM at
   producing Bangla at all.
2. **M-CLIPScore is language-blind.** It scores the 0%-Bangla system
   0.803 vs 0.546–0.552 for genuinely Bangla systems — reference-free
   multilingual-embedding scoring rewards semantic image match in the
   *wrong language*. BERTScore-bn (0.690 vs 0.747–0.758) is only
   mildly penalizing. Core evidence for the §C3 protocol
   recommendation: n-gram or human eval must anchor Bangla evaluation;
   M-CLIPScore alone is unusable for language fidelity.

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
