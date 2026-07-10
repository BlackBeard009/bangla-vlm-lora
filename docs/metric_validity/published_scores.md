# Published Bangla captioning scores — extraction for the metric-validity study

**Date:** 2026-07-11. Extracted by full-text reading where accessible;
rows marked UNVERIFIED could not be checked against the primary source.
Purpose: comparison targets for the no-vision trivial baselines
(`scripts/trivial_baselines.py`) and the §C3 metric-validity argument.

## Headline numbers per paper (best model each)

| Paper (venue, year) | Dataset (refs/img, origin) | Model | B1 | B2 | B3 | B4 | METEOR | CIDEr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Chittron (Procedia CS 2019) | BanglaLekha-orig (1, native) | VGG16+BiLSTM | — | — | — | "avg BLEU 2.5" sentence-level | — | — |
| Bornon (arXiv:2109.05218) | Bornon (5, native) | InceptionV3+Transformer | 0.696 | 0.589 | 0.507 | 0.439 | 0.361 | — |
| Bornon, same paper | BanglaLekha (2, native) | same | 0.665 | 0.556 | 0.476 | 0.408 | 0.255 | — |
| BAN-Cap (LREC 2022) | BAN-Cap (5, native) | AdaptiveAttn+CWR | 0.738 | 0.495 | 0.329 | 0.208 | 0.316 | 0.308 |
| CNN-Transformer (arXiv:2110.12442) | BanglaLekha (2, native) | ResNet101+Transformer | 0.694 | 0.580 | 0.505 | **2.22e-308** | 0.337 | — |
| BiCap (BLP-2025) | "Chitron" (≈1, native) | ResNet50+attn+LSTM | 0.714 | 0.593 | 0.464 | 0.327 | 0.37 | 0.29 |
| Masud (PLOS ONE 2025) | Flickr8k-BN (5, MT+fixes) | VGG19+LSTM+attn beam3 | 0.708 | 0.578 | 0.505 | 0.432 | 0.389 | — |
| Attn-CapBN (SSRN 2025, UNVERIFIED) | BanglaView (5, MT+edit) | CNN+attn | 0.623 | 0.487 | 0.394 | 0.333 | — | — |
| Attn-CapBN (UNVERIFIED) | BAN-Cap (5, native) | CNN+attn | 0.620 | 0.485 | 0.398 | 0.332 | — | — |
| IEEE 10441125 (ICCIT 2023) | Bornon+BAN-Cap | ViT enc-dec | UNVERIFIED — abstract has no numbers | | | | | |

Full per-paper tables (all Bornon configs, BAN-Cap Tables 2–5, Masud
greedy/beam grids, BiCap Table 2) preserved in the extraction notes —
regenerate via the research agent transcript if needed.

## Protocol chaos (evidence the numbers are not comparable)

1. **BLEU implementations/protocols never named.** Chittron: sentence-level
   single-ref (2.5/100). Bornon: corpus BLEU. BiCap: 0–100 scale.
   Palash et al.: abstract claims B2 0.630/B3 0.582, their own Table 1
   prints B2 0.580/B3 0.505 and **BLEU-4 = 2.22e-308** — unsmoothed
   zero 4-gram precision, published anyway, abstract contradicting body.
2. **Reference-count inflation.** BAN-Cap's own ablation: same model,
   same corpus, B1 0.738 with 5 refs → 0.480 with 2 refs. Papers with
   1, 2, and 5 refs are nonetheless compared in single tables (Bornon
   Table 4, BiCap Table 2).
3. **Vocabulary truncation + length caps.** Bornon: Keras top-5,000
   words. Masud: top-1,500 words, only 6–10-word captions used, max
   generated length 10. Both inflate n-gram overlap against short
   generic references.
4. **Cross-paper baseline rows don't reproduce.** BiCap's "Masud et al."
   row (B4 22.28) doesn't match Masud's own paper (B4 0.432 on a
   different corpus); provenance unexplained.
5. **Dataset identity confusion.** Chittron paper: 16,000 imgs × 1
   caption. Mendeley BanglaLekha release: 9,154 × 2. BiCap's "Chitron":
   15,438, citing a paper that doesn't exist in that form.

## The templated-output evidence (verbatim from the papers)

- BAN-Cap Fig. 6: model trained on BanglaLekha outputs **"একজন পুরুষ
  আছে।"** ("There is a man.") — 2.5/5 human score. Trained on Bornon:
  **"একটি ছেলে দাঁড়িয়ে আছে।"** — 1/5. Trained on BAN-Cap: "একজন পুরুষ
  ক্যামেরার দিকে তাকিয়ে হাসছে।" — 4.5/5.
- Bornon Fig. 15: **"একটি নৌকার নৌকার উপর কয়েকজন মানুষ আছে।"** ("people
  on a boat boat" — degenerate repetition), "একটি মেয়ে হাসতেছে।",
  "X দেখা যাচ্ছে" patterns. Also a figure gloss that doesn't match its
  own Bangla caption.
- Chittron Table III: "Water next to the water and some people are
  eating fish in the water."
- BAN-Cap's human eval of cross-dataset transfer: BanglaLekha-trained
  2.5/5, Bornon-trained 2.5/5, MT-data-trained **1.0/5** — i.e., the
  only prior human-judgment data in the field already says these
  systems produce template-grade captions.

## What this sets up

The trivial no-vision baselines (constant / random-train-caption /
frequency-words) scored with the same multi-ref protocol per corpus.
Hypothesis: they land within or near the published ranges above on
1–2-ref templated corpora, and far below our real models on BAN-Cap
5-ref. Combined with rows for our GiT-bridge and Qwen-QLoRA models,
the table tells the actual story: published n-gram scores on small
templated corpora measure corpus prior + protocol choices, not visual
captioning quality; meaningful Bangla captioning is possible (Qwen
QLoRA samples) but data, not architecture, is the ceiling.

Also directly usable in the paper:
- Chittron's own authors: BLEU "admittedly extremely poor," argue BLEU
  inappropriate with 1 ref — the field's founding paper already flagged
  the problem, then the field standardized on BLEU anyway.
- Our M-CLIPScore language-blindness result (0.803 for 0% Bangla
  output) extends the critique to reference-free metrics.
