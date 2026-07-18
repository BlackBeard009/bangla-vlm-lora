# No-vision trivial baselines vs real models — the actual story

**Date:** 2026-07-11 · `scripts/trivial_baselines.py`. All rows scored
with the identical multi-reference battery (pycocoevalcap BLEU/CIDEr,
BERTScore-bn, M-CLIPScore) on the identical BAN-Cap val split
(809 images × 5 native refs) as the real models.

Baselines derive from the train split only and NEVER see the image:
`constant` = most frequent training caption; `mode_len` = most frequent
caption of median length; `random` = random training caption per image;
`freq_words` = the 8 most frequent training tokens strung together
(the "pitfall words" string), same for every image.

## BAN-Cap val (5 refs, native)

| System | Sees image? | B1 | B2 | B3 | B4 | CIDEr | BERTScore-F1 | M-CLIPScore |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `constant` | no | 0.095 | 0.031 | 0.020 | 0.015 | 0.021 | 0.742 | 0.563 |
| `mode_len` | no | 0.198 | 0.077 | 0.034 | **0.020** | 0.042 | 0.752 | 0.539 |
| `random` | no | 0.178 | 0.046 | 0.009 | 0.000 | 0.020 | 0.732 | 0.542 |
| `freq_words` | no | **0.457** | 0.103 | 0.000 | 0.000 | 0.104 | **0.760** | 0.542 |
| GiT+bridge 5K | yes | 0.437 | 0.130 | 0.031 | 0.009 | 0.153 | 0.747 | 0.552 |
| GiT+bridge 10K | yes | 0.486 | 0.133 | 0.023 | 0.000 | 0.179 | 0.753 | 0.547 |
| GiT+bridge+img-LoRA | yes | 0.498 | 0.145 | 0.032 | 0.000 | 0.172 | 0.758 | 0.546 |
| Qwen2-VL-2B zero-shot | yes | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.690 | **0.803** |
| Qwen2-VL-2B QLoRA | yes | **0.532** | **0.331** | **0.177** | **0.082** | **0.300** | **0.804** | 0.561 |

## Findings

1. **A bag of frequent words beats real vision models on BLEU-1 and
   BERTScore.** `freq_words` (identical no-vision string for all 809
   images) outscores GiT-5K on BLEU-1 (0.457 vs 0.437) and ALL GiT
   runs on BERTScore-F1 (0.760 vs ≤0.758). A single template caption
   (`mode_len`) beats every GiT run on BLEU-4.
2. **Our own GiT rows do not clear the no-vision band on n-gram
   metrics.** Honest statement: the GiT-bridge motivation chain proves
   the vocabulary mechanism, but its BAN-Cap captions are only
   corpus-prior-plus-nouns — and the metrics show it once you add the
   trivial rows. Any paper reporting similar numbers without trivial
   baselines cannot distinguish its model from a frequency table.
3. **Only the Qwen QLoRA row clears every trivial baseline on every
   metric.** Margin is largest on CIDEr (0.300 vs ≤0.104) — tf-idf
   weighting resists common-word gaming. CIDEr is the only n-gram
   metric with meaningful discrimination here; BLEU-1 and BERTScore-bn
   are effectively saturated by corpus statistics.
4. **Metric indictment so far:** M-CLIPScore is language-blind (0.803
   for 0% Bangla), BERTScore-bn is template-blind (0.760 for a word
   salad), BLEU-1 is frequency-blind (0.457 for the same). The §C3
   protocol recommendation writes itself: CIDEr + human eval, with
   trivial-baseline rows mandatory.
5. This is BAN-Cap — 5 diverse native refs, the HARD target. Published
   Bangla results mostly live on BanglaLekha-family corpora (1–2
   templated refs, truncated vocabularies, capped lengths), where the
   trivial band sits higher still. BanglaLekha trivial rows: pending
   (images downloading). Combined with the published-scores extraction
   (`published_scores.md`): B4 0.408–0.439 reported there exceeds
   anything BAN-Cap's own authors achieved with 5 refs (0.208) —
   protocol, not model quality, is the likeliest explanation.
6. **Constructive reading:** meaningful Bangla captioning IS possible
   (Qwen QLoRA samples show real grounded composition) — the ceiling
   is data (7,282 train images → 494-token output vocabulary, and
   top-p sampling changes neither metrics nor vocabulary, ruling out
   decode as the cause). Scale experiment: BanglaView (31,783 imgs)
   curriculum, next.

## BanglaLekha val (2 refs, native, templated corpus) — completed 2026-07-18

Same baselines, BanglaLekha protocol (2 references, 915 val images),
now with the full battery (rerun of the 2026-07-11 n-gram-only pass;
all n-gram values reproduce exactly) **plus the missing comparison
row**: our GiT-bridge `banglalekha_full` adapter under the identical
protocol. The adapter was retrained locally 2026-07-18
(`configs/banglalekha_full_local.yaml`, same recipe + seed as the
Colab run; final val CE 2.561 vs Colab's 2.55 — cross-platform
reproduction) because the original lives only on Google Drive. Scored
via `configs/score_captions_banglalekha.yaml`; GiT row below is the
`min8_no_early_stop` default decode (BERTScore-F1 shown for
`baseline_beam4`, its best variant, in parentheses).

| System | Sees image? | B1 | B2 | B3 | B4 | CIDEr | BERTScore-F1 | M-CLIPScore |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `constant` (3 words, repeated) | no | 0.286 | 0.200 | 0.093 | 0.057 | 0.199 | **0.798** | 0.497 |
| `mode_len` (1 template) | no | 0.388 | 0.257 | **0.128** | **0.061** | 0.173 | 0.742 | 0.557 |
| `random` | no | 0.241 | 0.111 | 0.041 | 0.018 | 0.050 | 0.745 | **0.558** |
| `freq_words` | no | **0.471** | 0.217 | 0.033 | 0.000 | 0.176 | 0.738 | 0.546 |
| GiT+bridge `banglalekha_full` | yes | 0.465 | 0.080 | 0.015 | 0.000 | **0.279** | 0.715 (0.745) | 0.530 |

Findings, now with all three triplet rows in hand:

- **The vision model clears the no-vision band ONLY on CIDEr**
  (0.279 vs ≤0.199). On BLEU-1 it loses to `freq_words` (0.465 vs
  0.471), on BLEU-2..4 it loses to a single repeated template, and on
  BERTScore-F1 the constant 3-word caption beats it by 5 points
  (0.798 vs 0.745). Same shape as the BAN-Cap table: CIDEr is the
  only metric with discrimination on templated corpora.
- **The constant caption's BERTScore-F1 0.798 is the single most
  damning number in the study** — a no-vision 3-word string scores
  within 0.006 of the Qwen QLoRA system's 0.804 on BAN-Cap. BERTScore
  absolute values carry ~no signal across these corpora.
- **The no-vision floor covers ~68% of published BLEU-1** (0.471 vs
  0.665–0.694) and CIDEr 0.199 with a repeated 3-word caption.
- **Our honest vision model covers only ~68% of published BLEU-1
  too** (0.465 vs 0.665–0.694) — i.e., published BanglaLekha B1 sits
  ~0.2 above BOTH our trained model and the trivial floor, while
  published B4 (Bornon 0.408) exceeds our model's 0.000 by the full
  scale. Either those pipelines are far better than a bridged
  GiT-base — or (per the protocol-chaos notes) vocabulary truncation,
  length caps, and corpus-BLEU choices inflate the numbers. The
  triplet cannot decide which; the human-eval row can.
- **Palash et al. (arXiv:2110.12442) sits BELOW the trivial floor on
  BLEU-4**: their table prints 2.22e-308 vs the no-vision 0.061 —
  while their abstract claims B2 0.630. One published system is
  distinguishable from a template only in the wrong direction.
- **Bornon (B4 0.408) and TextMage (B4 0.238) exceed the floor 4–7×**
  — the floor alone does not invalidate them. The remaining questions
  for those rows are protocol ones: Keras top-5,000-word truncation
  (applied to refs and hypotheses), corpus BLEU on 2 refs producing
  B4 double what BAN-Cap's authors achieved with 5 refs, and the
  field's only human eval scoring this corpus family's outputs 2.5/5.
- Honest framing for the paper: the trivial band bounds how much of a
  published score is corpus prior; scores near the band are
  uninformative, scores far above it (Bornon) require protocol-level
  scrutiny rather than dismissal.
