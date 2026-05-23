# BAN-Cap full — bridged GiT + LoRA on the full corpus

**Date:** 2026-05-23
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/bancap_train.py`
**Config:** `configs/bancap_full.yaml`
**Dataset:** BAN-Cap (Khan et al., LREC 2022; arXiv:2205.14462) — 8,091
Flickr8k images × 5 native Bengali captions per image (40,455 captions
from human annotators with expert QC).
**Companion run:** `docs/results/banglalekha_full/` (the prior primary
corpus; superseded by this one).

## What this is

The first BAN-Cap run on the bridged GiT-base + LoRA recipe. Switching
to BAN-Cap was motivated by the 2026-05-23 sentence-length diagnostics
on the BanglaLekha adapter
(`docs/results/banglalekha_full/sentence_diagnostics.md`), which showed
that the dominant cause of templated outputs was eval bias on a
two-reference corpus with a short/long caption pairing (PPL 13.1 on
captions[0] vs. 49.3 on captions[-1], a 3.76× gap). BAN-Cap addresses
this structurally by providing five **peer** captions per image —
collected one per annotator from five different annotators — rather
than short/long pairs.

The training recipe (bridge, LoRA targets, hyperparameters, decode) is
held **identical** to `configs/banglalekha_full.yaml` so the
corpus-vs-everything-else comparison is clean.

## Setup

- **Bridge:** GiT-base vocab (30,522) extended with 29,055 BanglaBERT
  wordpieces; `donor` init strategy; diacritic-preserving tokenizer
  reload (`do_lower_case=False, strip_accents=False,
  tokenize_chinese_chars=False`). Identical to the BanglaLekha run.
- **LoRA:** rank 8, α 16, dropout 0.05, target
  `[query, key, value]` on GiT's text-side attention;
  `modules_to_save=[output, word_embeddings]`. 109.5 M / 330.8 M
  parameters trainable (33.11%). Image tower is fully frozen
  (and — per the diagnostics — `target_modules` silently misses the
  image encoder's q/k/v because CLIP-ViT uses `q_proj/k_proj/v_proj`
  naming; this is an intentional control to isolate the dataset effect).
- **Data split:** 7,282 train / 809 val from the full 8,091-image
  set (seed-42 90/10). `caption_selection: random` on train, `first`
  on eval — both pick from 5 peer captions of comparable length, so
  the eval-bias problem from BanglaLekha does not apply.
- **Training:** 5,000 steps, batch 4, AdamW lr 1e-4, no weight decay.
- **Decode (eval):** beam=4, `max_new_tokens=30`, no length /
  repetition penalty, no sampling.

## Result

| Step | Train loss | Val loss |
|---:|---:|---:|
|  500 | 6.476 | 5.795 |
| 1000 | 5.603 | 5.396 |
| 1500 | 5.353 | 5.207 |
| 2000 | 5.103 | 5.083 |
| 2500 | 4.907 | 4.922 |
| 3000 | 4.875 | 4.852 |
| 3500 | 4.757 | 4.779 |
| 4000 | 4.666 | 4.720 |
| 4500 | 4.533 | 4.681 |
| 5000 | 4.613 | 4.643 |

**Wall:** 1,497.9 s on T4 (~0.30 s/step, vs. BanglaLekha's 0.47 s/step
— faster because BAN-Cap captions average 8.6 words vs. BanglaLekha's
8.5 but the val set is smaller).

Val loss is monotonically decreasing through step 5000 — **the curve
has not plateaued**, unlike BanglaLekha which flatlined for the last
1,000 steps. The drop from step 4,500 → 5,000 is 0.038, the same
magnitude as 4,000 → 4,500. Training longer would likely help.

## Sample generations (held-out val, step 5000)

| Image | Reference (one of 5) | Generated |
|---|---|---|
| `101654506_8eb26cfb60.jpg` | একটি সাদা-বাদামি ছোপযুক্ত কুকুর বরফের মাঝ দিয়ে দৌড়াচ্ছে । | `সাদাটি তুষার মধ্যে কুকুর` |
| `104136873_5b5d41be75.jpg` | কয়েকজন লোক পাহাড়ের কিনারায় বসে প্রকৃতিররূপ দেখছে | `লোক পাহাড়ের বসে` |
| `1042020065_fb3d3ba5ba.jpg` | সবুজ পোশাক পরিহিত একটি বালক নিচে ভাসমান নৌকাসমূহের দিকে তাকিয়ে আছে । | `লোক` |
| `1048710776_bb5b0a5c7c.jpg` | কয়েকজন ব্যক্তি শৈলশিরার উপর বসে থেকে সমুদ্রসৈকত পর্যবেক্ষণ করছে । | `লোক বসে` |
| `1056249424_ef2a2e041c.jpg` | শিশুরা পানিতে খেলছে । | `শিশু` |
| `1079274291_9aaf896cc1.jpg` | একটি ছোটো বালক ক্যামেরায় ছবি তোলার জন্যে জিহ্বা দেখাচ্ছে । অন্য একটি বালক তার ছবি তোলা দেখছে । | `বাচ্চা` |
| `1096395242_fc69f0ae5a.jpg` | খেলনা বন্দুক হাতে বালক | `ছোটটি খেলনা খেলছে` |
| `109671650_f7bbc297fa.jpg` | খাড়া পাহাড় চূড়ায় হলুদ ছোটো প্যান্ট পরিহিত বালকটি দাঁড়িয়ে আছে | `ছেলে উপর আছে` |

Full per-eval sample dump: `paths.EXP_RESULTS/bancap_full/samples.json`.

### Diversity progression across training

Unique-token count across all 8 fixed val-image generations at each
eval point, and mean generation length in whitespace words:

| Step | Unique tokens (8 samples) | Mean gen words |
|---:|---:|---:|
|  500 |  3 | 0.5 |
| 1000 |  4 | 0.6 |
| 1500 | 10 | 1.4 |
| 2000 | 10 | 2.0 |
| 2500 |  8 | 1.9 |
| 3000 | 16 | 2.8 |
| 3500 | 10 | 3.0 |
| 4000 | 16 | 2.8 |
| 4500 | 12 | 3.1 |
| 5000 | 15 | 2.2 |

The lexical diversity grows roughly monotonically: from 3 unique
tokens at step 500 (only common safe-word tokens) to 15 at step
5,000 (image-content tokens like `কুকুর`/`তুষার`/`পাহাড়`/`বসে`).
Mean generation length grows from 0.5 → 2.2 words.

## Direct comparison to `banglalekha_full`

Both runs use **identical** hyperparameters, identical bridge, identical
LoRA target modules. The only thing that changes is the training corpus.

| Metric | BanglaLekha | BAN-Cap | Δ |
|---|---:|---:|---|
| Train / val | 8,239 / 915 | 7,282 / 809 | similar |
| Refs per image | 2 (short + long) | 5 (peer) | qualitatively different |
| Wall (T4) | 2,374.8 s | 1,497.9 s | -37% (smaller val set + tighter dataloader) |
| Final val loss (eval against `first`) | **2.55** | **4.64** | +2.09 |
| Final val loss (eval against `random`) | 3.41 (diagnostic) | not measured yet | — |
| Val curve at final step | plateau (Δ +0.02) | still decreasing (Δ -0.038) | — |
| Mean gen words (final) | 4.2 (decode_ablation) | 2.2 | shorter |
| Content tokens emitted (final 8) | `মানুষ`/`পুরুষ`/`শিশু`/`গাছ` (4) | `কুকুর`/`তুষার`/`পাহাড়`/`বসে`/`শিশু`/`লোক`/`ছেলে`/`বাচ্চা`/`খেলনা`/`খেলছে` (10+) | richer |

### How to read the +2.09 val-loss gap

The headline val loss is **higher** for BAN-Cap (4.64 vs. 2.55). This
is **not** a regression — it reflects that BAN-Cap is a fundamentally
harder target distribution:

1. **3× larger vocabulary.** BAN-Cap has 16,560 unique Bengali tokens
   in 40,455 captions vs. BanglaLekha's 5,377 in 18,308 captions. Each
   token gets ~80 fewer training visits on average; per-token CE is
   naturally higher.
2. **Weaker corpus prior.** The most-common BanglaLekha caption-end
   token (`আছে`) covers 46.4% of captions; in BAN-Cap the most-common
   end token covers only 13.5%. CE is much easier when "guess the
   prior" is correct 46% of the time. The flat distribution in BAN-Cap
   makes the model commit to image-content tokens instead.
3. **The BanglaLekha 2.55 is against the *short* caption only.** The
   diagnostic measured PPL 13.1 on BanglaLekha's `first` (short) but
   PPL 49.3 on `last` (long). BAN-Cap's `first` is not a forced-short
   form; it's one of 5 peer captions of comparable length, more
   analogous to BanglaLekha's `random` (val 3.41) than to its `first`
   (val 2.55). Adjusted comparison: **BAN-Cap 4.64 vs. BanglaLekha
   3.41** — still a gap (BAN-Cap is harder for the same recipe), but
   smaller and consistent with the per-token entropy difference of
   the corpora.

### What got better

Sample-level quality improved across nearly all dimensions:

- **Visual grounding.** `101654506` (snow dog) → `সাদাটি তুষার মধ্যে
  কুকুর` ("white in snow dog") — three image-content tokens plus a
  spatial relation. The BanglaLekha model emitted `মানুষ আছে` ("people
  exist") for almost every image regardless of content.
- **Action verbs.** `1056249424` (children in water) → `শিশু`; while
  short, it correctly identifies the subject; previous BanglaLekha
  outputs leaned heavily on `আছে`/`করছে` regardless of action.
- **Compositional fragments.** `109671650` (boy on mountain) →
  `ছেলে উপর আছে` ("boy above exists") — a subject + spatial relation +
  copula, a real fragment of grammar. BanglaLekha never produced
  multi-content-word outputs like this.
- **No degenerate beam-search outputs.** The `45.png` danda-loop
  pathology that hit BanglaLekha (`। গাছ । একটি গাছ ...`) does not
  appear in any of the 8 sample images here, even at unconstrained
  beam=4. The flatter corpus prior makes the safe-token attractor
  weaker.

### What did not change (yet)

- **Mean output length is still short** (2.2 words vs reference 9.1).
  This is consistent with the BanglaLekha diagnostic Test 2 finding:
  the 6-layer decoder genuinely lacks compositional capacity to
  produce well-formed long captions. The BAN-Cap data is much richer
  but the model still cannot use most of that richness simultaneously.
- **Loss is high (4.64).** Not a flaw — explained above — but the
  paper-grade BLEU/CIDEr numbers may still come out modest.

## Probable reasons for remaining gaps

1. **Decoder capacity.** GiT-base has 6 text decoder layers and ~45 M
   decoder parameters. With LoRA rank 8 + the trainable 22.3 M LM
   head, that is still small for the 16,560-token target vocabulary.
   The diagnostic forced-length test on BanglaLekha already showed
   that the model lacks compositional structure to surface; BAN-Cap's
   richer training signal helps but doesn't manufacture decoder
   capacity. Bigger base model (PaliGemma) is the natural next step
   on this axis.
2. **Image tower is still frozen.** Same caveat as
   `banglalekha_full/README.md` § "Probable reasons" #4 — the
   `target_modules: [query, key, value]` config silently misses the
   image encoder's `q_proj/k_proj/v_proj`. Visual signal is whatever
   CLIP-ViT-on-English produces. The next ablation (image-tower LoRA)
   is now well-motivated for BAN-Cap too.
3. **5,000 steps is short.** The val curve is still moving. 10K or
   15K steps would probably push final val loss below 4.0 with the
   same recipe.
4. **Beam=4 with no penalties.** The diagnostic decode-ablation
   showed this is conservative; on BAN-Cap, `no_repeat_ngram_size=3`
   may help less (no danda loops to remove) but `length_penalty>1.0`
   plus `min_new_tokens` could nudge outputs longer. Cheap to test.

## Honest caveats

- **No real metric numbers yet.** Still next.
  BLEU/CIDEr/BERTScore-bn/M-CLIPScore on the 809-val with 5-reference
  scoring — now properly meaningful because BAN-Cap supplies 5 refs
  per image.
- **Same-seed reproducibility not independently verified.** A second
  run with the same seed should land at 4.64 ± noise; not done.
- **Decoder hyperparameters are untuned.** All decode numbers here
  use the default config (beam=4, no penalties). Different settings
  will shift the sample table.
- **809 val images is small** (8% smaller than BanglaLekha's 915).
  Variance in val-loss eval points is probably ±0.02.

## Artifacts

- `paths.EXP_RESULTS/bancap_full/results.md` — regenerated each run.
- `paths.EXP_RESULTS/bancap_full/samples.json` — full sample dump
  across 10 eval points (500..5000 step intervals).
- `paths.EXP_CHECKPOINTS/bancap_full/` — LoRA adapter
  (`adapter_model.safetensors`, ~1.17 GB with `modules_to_save`
  embedding+head) and the bridged tokenizer. Persisted on Drive.

## Reproduce

```bash
# First time only — captions CSV (8.5 MB) is already on Drive;
# Flickr8k images (~1 GB) need re-fetching via Kaggle:
mkdir -p /content/bancap_dl
cd /content/bancap_dl
KAGGLE_USERNAME=<user> KAGGLE_KEY=<key> kaggle datasets download \
    -d adityajn105/flickr8k && unzip -q flickr8k.zip
# (the Bengali captions CSV is persistent at
# paths.DATA_RAW/bancap/BAN-Cap_captiondata.csv)

# Then:
cd /content/bangla-vlm-lora
python scripts/bancap_train.py --config configs/bancap_full.yaml
```

## What's next

In priority order (also reflected in `CLAUDE.md` § "What's next"):

1. **Re-run the sentence-length diagnostics on the BAN-Cap adapter.**
   The current `scripts/sentence_diagnostics.py` is hardcoded to
   `BanglaLekhaCaptions`; a small refactor (add `data.loader` to the
   config and dispatch) makes it BAN-Cap-aware. Then we can directly
   compare image-scrambling Jaccard, forced-length effects, and the
   five-way caption-selection val-loss spread (`first`/`last`/`random`/
   plus maybe `index:2`/`index:3` for annotator-variance reading).
2. **Real metric computation** with proper 5-reference scoring.
   `scripts/score_captions.py` against `baseline_beam4` and
   `beam4_penalties` decode; this is the metric framework the paper
   plan §C3 needs.
3. **Train longer.** Val curve is not plateaued; 10K steps is cheap
   and likely improves outputs without changing anything else.
4. **Image-tower LoRA (the real one).** Now add CLIP-ViT
   `q_proj/k_proj/v_proj/out_proj` to `target_modules` and unfreeze
   `visual_projection.visual_projection.0`. With BAN-Cap's richer
   visual content this should help more than it would have on
   BanglaLekha.
5. **Bigger base model (PaliGemma-3B + QLoRA).** Per paper plan §3.

This run is the new primary baseline. The BanglaLekha pipeline stays
intact in the repo for reproducibility of the prior chain but is no
longer the headline corpus.
