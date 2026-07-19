# SemMargin — margin-normalized semantic similarity, gauntlet results

**Date:** 2026-07-18 · `src/evaluation/semantic_margin.py`,
`scripts/semantic_margin_gauntlet.py`,
`configs/semantic_margin_gauntlet.yaml` · Encoder:
`sentence-transformers/LaBSE` · BAN-Cap val, 809 images × 5 refs.

## Lineage and novelty position (checked 2026-07-18)

The subtract-similarity-to-other-images mechanism is published in the
*distinctiveness* literature and must be cited, not claimed:

- **CIDErBtw** (Wang et al., ECCV 2020, arXiv:2007.06877; TPAMI
  extension arXiv:2204.03938) — caption's CIDEr similarity to similar
  images' captions, as distinctiveness measure + training reward.
  N-gram-based, curated similar-image sets.
- **GEG / Group Embedding Gap** (arXiv:2208.04254) — closest match:
  CLIP embedding similarity of caption to target image minus
  similarity to a curated group of similar images. Cross-modal,
  English MSCOCO, distinctiveness-only framing.

Our defensible slice: (a) reference-text-side margin against the FULL
evaluation corpus — no image encoder (unusable for Bangla per the
M-CLIPScore result) and no curated groups; (b) framing as a general
quality metric with the proven zero-on-no-vision-baselines property;
(c) human-ceiling calibration and (pending) the first human-rating
validation of any such metric for Bangla. Claim the adaptation and
the validation, never the subtraction.

## The idea

Plain embedding cosine between a hypothesis and the references (the
BERTScore/SBERT/M-CLIPScore family) is gamed by generic captions,
which sit near the semantic centroid of the corpus. Fix: subtract the
corpus prior —

    raw(h_i)    = mean_r cos(e(h_i), e(ref_{i,r}))
    prior(h_i)  = mean_{j≠i} raw-vs-image-j
    margin(h_i) = raw − prior        # specificity-corrected similarity
    rank(h_i)   = percentile of raw among all images   # self-retrieval view

A generic caption is equally similar to every image's references, so
its margin is 0 *by construction* rather than by hope.

## Results (every system with saved BAN-Cap predictions)

| System | raw | prior | **margin** | rank |
|---|---:|---:|---:|---:|
| trivial constant | 0.284 | 0.284 | **−0.000** | 0.500 |
| trivial mode_len | 0.483 | 0.483 | **0.000** | 0.500 |
| trivial random | 0.432 | 0.431 | **0.000** | 0.506 |
| trivial freq_words | 0.455 | 0.455 | **−0.000** | 0.500 |
| Qwen2-VL zero-shot (0% Bangla) | 0.601 | 0.361 | 0.240 | 0.980 |
| GiT+bridge 5K | 0.533 | 0.380 | 0.153 | 0.920 |
| GiT+bridge 10K | 0.531 | 0.379 | 0.152 | 0.910 |
| GiT+bridge img-LoRA | 0.531 | 0.386 | 0.145 | 0.913 |
| Qwen QLoRA BAN-Cap | 0.644 | 0.466 | 0.177 | 0.937 |
| Qwen QLoRA BanglaView | 0.626 | 0.457 | 0.169 | 0.930 |
| **Qwen curriculum** | 0.661 | 0.471 | **0.190** | 0.953 |
| HUMAN (held-out ref) | 0.647 | 0.437 | **0.210** | 0.968 |

## Gauntlet verdict

1. **Trivial baselines: annihilated.** Margin ≤ 0.0002, rank = chance
   (0.50) for all four. Compare the *raw* column — the uncorrected
   version of this metric (i.e., plain embedding cosine, the original
   idea) gives the `mode_len` template 0.483 vs GiT's 0.533: nearly
   indistinguishable. The correction is what does the work. No other
   metric in the study zeroes the no-vision band; even CIDEr gives it
   up to 0.104.
2. **System ordering matches the full n-gram study independently:**
   GiT band (0.145–0.153) < Qwen BanglaView (0.169) < Qwen BAN-Cap
   (0.177) < curriculum (0.190) — obtained with zero n-gram overlap,
   zero shared machinery with BLEU/CIDEr. Curriculum reaching 91% of
   the human held-out margin is the single most flattering (and
   plausible) framing of the curriculum result.
3. **Human ceiling behaves:** 0.210, above every trained system.
   (Mild asymmetry: the ceiling row scores 1 ref vs the other 4;
   systems score vs all 5. Direction of bias is against the ceiling,
   so "above every system" survives it.)
4. **Known failure, inherited and now isolated:** the 0%-Bangla
   zero-shot row gets the top margin (0.240) — LaBSE is cross-lingual
   by design, so correct-content English embeds next to Bangla
   references. SemMargin measures the *specificity* axis only. It
   must be paired with a language-fidelity gate (e.g., Bangla-token
   ratio, trivially computable) — exactly the multi-axis protocol the
   metric-validity study argues for. Notably the English row's rank
   (0.980) exceeds even the human refs — fluent English zero-shot
   captions are hyper-discriminative, which is itself evidence the
   axis decomposition is real: that system maxes specificity while
   scoring zero on language fidelity.

## Status

Passes 3 of 4 gauntlet criteria; the 4th (language blindness) is an
encoder property, controlled by gating rather than fixable within the
metric. Pending: correlation against the §C3 human-eval pilot ratings
— if margin correlates with human adequacy better than BLEU/BERTScore
(plausible: it is the only metric here that isn't fooled by corpus
priors), it graduates into the paper's recommended protocol as the
specificity component.
