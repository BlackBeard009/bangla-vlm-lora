# Bornon reproduction — do the published BanglaLekha numbers reproduce?

**Date:** 2026-07-18 · `scripts/bornon_repro.py` +
`configs/bornon_repro.yaml` · Target: Bornon (Shah et al.,
arXiv:2109.05218), Table 2, best BanglaLekha row — **corpus BLEU
0.665 / 0.556 / 0.476 / 0.408**, the strongest published number on the
corpus and the one that most exceeds both our trivial floor (B4 0.061)
and our honest GiT row (B4 ≈ 0).

## Why reproduce instead of cite

No code or weights were released (checked 2026-07-18: no repository
link in the arXiv page, paper body, or the group's public profiles).
The published-scores extraction (`published_scores.md`) documents
protocol red flags across the field — vocabulary truncation, unnamed
BLEU implementations, 2-ref corpus BLEU exceeding 5-ref results from
better-resourced teams — but flags are not evidence of inflation. A
faithful recipe reproduction converts suspicion into a measurement:
train their stated pipeline, then score the SAME model under (1) their
protocol, (2) the same scorer without vocabulary truncation, and
(3) our honest battery used for every other row in this study. The
protocol deltas are then facts about measurement, not accusations
about a specific paper's honesty.

## What the paper states (used as-is)

| Item | Paper's spec |
|---|---|
| Image encoder | InceptionV3, softmax removed, 299×299 → 8×8×2048, precomputed |
| Text pipeline | Keras text tokenizer, **top-5,000 words**, zero-padded to max length |
| Architecture | Vaswani encoder-decoder; image tokens → encoder, caption tokens → decoder; padding + look-ahead masks |
| Best BanglaLekha config | 3 layers, 1 head (Table 2) |
| Training | 50 epochs, batch 64, Adam, Vaswani LR schedule (warmup 4000), SparseCategoricalCrossentropy |
| Split | 7,154 train / 1,000 val / 1,000 test (Table 1) |
| Eval | Corpus BLEU-1..4 (Eq. 7–10; standard modified n-gram precision), 2 refs |

## What the paper does NOT state (our assumptions)

- **d_model / dff / dropout** → 512 / 2,048 / 0.1 (Vaswani defaults;
  the paper's wording follows the TensorFlow tutorial lineage).
- **Decode method** → greedy (beam search discussed only for prior
  work; TF tutorial decodes greedily).
- **Start/end sentinels** → `startseq`/`endseq` words (Keras default
  filters strip `<`/`>`, so `<start>` tokens cannot survive their
  stated tokenizer; `startseq`/`endseq` is the standard Keras-tutorial
  workaround).
- **BLEU implementation** → NLTK `corpus_bleu` (the overwhelmingly
  common choice in Keras-era captioning code; the paper's Eq. 7–10 is
  exactly NLTK's algorithm).
- **Whether references were round-tripped through the top-5,000
  tokenizer** → unknowable; we report BOTH variants — that ambiguity
  is itself one of the study's findings.
- Seed / split assignment → our seed 42. Their exact split is
  unrecoverable; with 1,000 test images, split noise on corpus BLEU
  is small relative to the deltas of interest.

Fidelity caveat for the paper: this is a *recipe* reproduction under
documented assumptions, not a rerun of their artifact. Conclusions are
drawn from protocol deltas on our trained model, which are valid
regardless of whether our checkpoint matches theirs — plus a
did-it-reproduce comparison stated with the assumption list attached.

## Results (2026-07-18, 50 epochs, 769 s on the 4060 Ti)

Train loss 8.5 → 0.364 (fully converged, likely memorizing — as 50
epochs on 7,154 images must). Greedy decode of the 1,000-image test
split: mean 6.7 words, 0 empty, 623 unique output tokens.

| Scoring protocol | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| Published (their Table 2) | 0.665 | 0.556 | 0.476 | **0.408** |
| Ours, their protocol (top-5000 refs+hyps, NLTK) | 0.374 | 0.206 | 0.111 | **0.064** |
| Ours, same scorer, no truncation | 0.367 | 0.203 | 0.109 | 0.063 |
| Ours, honest battery (pycocoevalcap) | 0.452 | 0.288 | 0.172 | 0.106 |

Honest-battery extras: CIDEr **0.495**, BERTScore-F1 0.806.

### Findings

1. **The published numbers do not reproduce.** The paper's own stated
   recipe, trained to full convergence and scored under the paper's
   own stated protocol, lands at B4 0.064 — a **6.4× gap** to the
   published 0.408. No scoring variant we tested closes it: the most
   generous scorer (pycocoevalcap) reaches 0.106, still 4× short.
   With the assumption list attached (d_model, decode, seed, split
   assignment), the honest claim is: *the result is not reproducible
   from the information the paper provides.*
2. **The top-5,000 truncation is NOT the inflation mechanism here.**
   BanglaLekha's fitted vocabulary is only 5,075 words, so the cap
   drops 75 rare words: B1 moves 0.367 → 0.374. Our prior suspicion
   that vocabulary truncation drives the BanglaLekha-family numbers is
   wrong for this corpus (it may still matter for Masud's top-1,500
   cap on Flickr8k-BN). Honest study kills its own hypothesis.
3. **The scorer implementation is worth up to +68% on B4.**
   Identical predictions, identical references: NLTK corpus BLEU
   B4 0.063 vs pycocoevalcap 0.106, B1 0.367 vs 0.452. Papers that
   never name their BLEU implementation (all of them, per
   `published_scores.md`) have a ±40–70% free parameter on every
   reported number.
4. **The remaining 4–6× gap needs a non-protocol explanation.**
   Candidates we cannot test without their artifacts: evaluation on
   train images or the internal-validation split, sentence-level BLEU
   averaged then reported as corpus BLEU, a scoring-time reference
   handling bug, or an unstated architectural/data difference. What we
   CAN say: reference-count inflation (BAN-Cap's own 5→2-ref ablation,
   0.738→0.480 B1) and implementation choice explain large fractions
   of cross-paper spread elsewhere in the table; neither explains
   Bornon's B4.
5. **The recipe itself is respectable under honest scoring** — and
   that's a finding FOR the field: CIDEr 0.495 clears the trivial
   floor (0.199) by 2.5×, beats our GiT-bridge BanglaLekha row
   (0.279), and B4 0.106 (pycoco) is the best honest BanglaLekha
   number in this study. A 12M-param from-scratch transformer with
   precomputed InceptionV3 features is a genuinely decent baseline;
   the published 0.408 was never needed to justify the architecture.

### Comparison rows now available (BanglaLekha 2-ref, honest battery)

| System | B1 | B4 | CIDEr | BERTScore-F1 |
|---|---:|---:|---:|---:|
| Trivial floor (best of 4) | 0.471 | 0.061 | 0.199 | 0.798 |
| GiT+bridge `banglalekha_full` | 0.465 | 0.000 | 0.279 | 0.745 |
| Bornon-recipe reproduction | 0.452 | **0.106** | **0.495** | **0.806** |

The reproduction becomes the best honest system row on BanglaLekha —
useful both as a stronger comparison point and as evidence that our
scoring harness is not simply hostile to small models.
