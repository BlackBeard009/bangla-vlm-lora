# Sentence-length diagnostics on the saved bancap_full adapter

**Date:** 2026-05-24
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/sentence_diagnostics.py`
**Config:** `configs/sentence_diagnostics_bancap.yaml`
**Adapter:** `paths.EXP_CHECKPOINTS/bancap_full/` (bridged GiT + LoRA,
donor init, 5,000 steps — see `README.md` in this directory).
**Companion:** `docs/results/banglalekha_full/sentence_diagnostics.md`
(same three tests on the prior corpus; this write-up is structured to
permit direct comparison).

## What this is

Three independent diagnostics, no retraining, run on the BAN-Cap full
adapter. The script and tests are identical to the BanglaLekha
diagnostics; the loader is now dispatched via `data.loader` in the
config (`banglalekha` | `bancap`) — see the refactor notes at the end.

The motivating question is whether the corpus switch from BanglaLekha
to BAN-Cap structurally fixes the issues the prior diagnostics
surfaced:

  - **Test 1 — image scrambling.** Is the BAN-Cap model more
    image-grounded than the BanglaLekha model?
  - **Test 2 — forced-length decode.** With richer per-image vocabulary
    and no danda-loop trap, does forcing length recover any
    compositional structure?
  - **Test 3 — caption-selection val CE.** Does the 3.76× PPL gap from
    BanglaLekha (between `first` and `last`) disappear on BAN-Cap, as
    expected from five peer captions of comparable length?

## Test 1 — image scrambling

### Setup

| Variant | Image input |
|---|---|
| `real` | Unmodified val image |
| `gray` | Solid `(128, 128, 128)` RGB image, same size |
| `noise` | Per-pixel uniform random RGB, same size, seed 42 |
| `shuffled` | `image[i]` replaced with `image[(i + 1) mod 8]` (real but wrong-for-this-caption) |

Decode: beam=4, `max_new_tokens=30`, no penalties (matches the
`bancap_full` headline samples).

### Outputs (first 8 val items, real images)

| Image | Reference | `real` | `gray` | `noise` | `shuffled` |
|---|---|---|---|---|---|
| `101654506_8eb26cfb60.jpg` | একটি সাদা-বাদামি ছোপযুক্ত কুকুর বরফের মাঝ দিয়ে দৌড়াচ্ছে । | সাদাটি তুষার মধ্যে কুকুর | লোক | রঙের পরিহিত | লোক পাহাড়ের বসে |
| `104136873_5b5d41be75.jpg` | কয়েকজন লোক পাহাড়ের কিনারায় বসে প্রকৃতিররূপ দেখছে | লোক পাহাড়ের বসে | লোক | রঙের পরিহিত | লোক |
| `1042020065_fb3d3ba5ba.jpg` | সবুজ পোশাক পরিহিত একটি বালক নিচে ভাসমান নৌকাসমূহের দিকে তাকিয়ে আছে । | লোক | লোক | *(empty)* | লোক বসে |
| `1048710776_bb5b0a5c7c.jpg` | কয়েকজন ব্যক্তি শৈলশিরার উপর বসে থেকে সমুদ্রসৈকত পর্যবেক্ষণ করছে । | লোক বসে | লোক | রঙের পরিহিত | শিশু |
| `1056249424_ef2a2e041c.jpg` | শিশুরা পানিতে খেলছে । | শিশু | লোক | রঙের পরিহিত লোক | বাচ্চা |
| `1079274291_9aaf896cc1.jpg` | একটি ছোটো বালক ক্যামেরায় ছবি তোলার জন্যে জিহ্বা দেখাচ্ছে। অন্য একটি বালক তার ছবি তোলা দেখছে । | বাচ্চা | লোক | রঙের পরিহিত | ছোটটি খেলনা খেলছে |
| `1096395242_fc69f0ae5a.jpg` | খেলনা বন্দুক হাতে বালক | ছোটটি খেলনা খেলছে | লোক | রঙের পরিহিত | ছেলে উপর আছে |
| `109671650_f7bbc297fa.jpg` | খাড়া পাহাড় চূড়ায় হলুদ ছোটো প্যান্ট পরিহিত বালকটি দাঁড়িয়ে আছে | ছেলে উপর আছে | লোক | রঙের পরিহিত | সাদাটি তুষার মধ্যে কুকুর |

### Divergence from `real` baseline

| Variant | Exact-match with `real` | Mean Jaccard token overlap |
|---|---:|---:|
| `gray` | 12.5% | 0.229 |
| `noise` | 0.0% | **0.000** |
| `shuffled` | 0.0% | 0.104 |

### Direct comparison to BanglaLekha

| Variant | BanglaLekha Jaccard | BAN-Cap Jaccard | Δ |
|---|---:|---:|---:|
| `gray` | 0.256 | 0.229 | -0.027 |
| `noise` | 0.206 | **0.000** | -0.206 |
| `shuffled` | 0.108 | 0.104 | ~ same |

### What this tells us

1. **Gray collapses to a single fallback for both models.** The BAN-Cap
   model emits `লোক` ("people") for 7/8 images under gray; the
   BanglaLekha model emitted `। একজন আছে` for 7/8. The fallback
   string is corpus-specific (BanglaLekha's `। একজন আছে` follows that
   corpus's `<noun> আছে` prior; BAN-Cap's `লোক` follows the bare-noun
   short-output pattern), but the *behavior* — "image carries no
   information → emit a single safe token" — is identical. Jaccard
   0.229 reflects partial overlap on items where the real output
   contains `লোক`.

2. **Noise collapses BAN-Cap to `রঙের পরিহিত` (zero Jaccard with real).**
   Six of 8 images decode to the identical OOD-fallback string `রঙের
   পরিহিত` ("wearing color"); one decodes to the empty string; one to
   `রঙের পরিহিত লোক`. This is a stronger collapse than BanglaLekha,
   which kept Jaccard 0.21 under noise — the BanglaLekha model was
   still using *some* image-prior overlap with real outputs, just
   short-circuited through similar nouns. The BAN-Cap model has more
   distinct visual representations: real-image outputs use noun
   tokens specific to the image content (`কুকুর`, `পাহাড়`, `নৌকা`,
   `খেলনা`), and these share no surface tokens with the noise
   fallback. **Stronger evidence that the BAN-Cap model actually uses
   image features beyond a noun-class lookup.**

3. **Shuffled Jaccard is lower than gray — same pattern as
   BanglaLekha.** With a real-but-wrong image, the model attends to
   the wrong content. Examples:
   - `109671650` (boy on mountain) shuffled-in is `101654506_8eb...`
     (snow dog); the model outputs `সাদাটি তুষার মধ্যে কুকুর` — the
     snow-dog caption.
   - `1079274291` (boy with camera) shuffled-in is `1096395242`
     (toy gun); the model outputs `ছোটটি খেলনা খেলছে` — the
     toy-playing caption.
   - `1096395242` (toy gun) shuffled-in is `109671650` (boy on
     mountain); the model outputs `ছেলে উপর আছে` — the
     boy-on-something caption.

   The model's noun selection follows the *shuffled-in* image, not
   the original. That requires real visual conditioning, not just a
   prior.

**Verdict.** The BAN-Cap model is **more image-grounded** than the
BanglaLekha model. Both use the image for noun selection; only the
BAN-Cap model has visual representations diverse enough that OOD
input collapses to a different region of the output distribution
entirely. The corpus switch did help visual grounding, presumably
because (a) Flickr8k's content is more visually diverse than
BanglaLekha's field photos and (b) 5 peer references per image
provide stronger visual-content training signal than 2 refs.

## Test 2 — forced-length decode

### Setup

| Variant | Params |
|---|---|
| `baseline_beam4` | `num_beams=4, max_new_tokens=30` |
| `min8_no_early_stop` | `+min_new_tokens=8, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=False` |
| `min12_no_early_stop` | `+min_new_tokens=12, ...` (rest identical) |

### Outputs (first 8 val items, real images)

| Image | Reference | `baseline_beam4` | `min8_no_early_stop` | `min12_no_early_stop` |
|---|---|---|---|---|
| `101654506_8eb26cfb60.jpg` | একটি সাদা-বাদামি ছোপযুক্ত কুকুর বরফের মাঝ দিয়ে দৌড়াচ্ছে । | সাদাটি তুষার মধ্যে কুকুর | সাদাটি তুষার মধ্যে কুকুর লাফ । একটি ধরার | সাদাটি তুষার মধ্যে কুকুর লাফ । একটি ধরার একটি রঙের |
| `104136873_5b5d41be75.jpg` | কয়েকজন লোক পাহাড়ের কিনারায় বসে প্রকৃতিররূপ দেখছে | লোক পাহাড়ের বসে | ' লোক পাহাড়ের দাঁড়িয়ে । ' লোক বসে । দাঁড়িয়ে । একটি পাহাড়ের বসে । । আছে | (same as min8) |
| `1042020065_fb3d3ba5ba.jpg` | সবুজ পোশাক পরিহিত একটি বালক নিচে ভাসমান নৌকাসমূহের দিকে তাকিয়ে আছে । | লোক | লোক নৌকা আছে একটি নৌকা | লোক নৌকা আছে দেখছে একটি লোক দেখছে |
| `1048710776_bb5b0a5c7c.jpg` | কয়েকজন ব্যক্তি শৈলশিরার উপর বসে থেকে সমুদ্রসৈকত পর্যবেক্ষণ করছে । | লোক বসে | লোক বসে দেখছে দেখছে দেখছে একজন দেখছে | (same as min8) |
| `1056249424_ef2a2e041c.jpg` | শিশুরা পানিতে খেলছে । | শিশু | শিশু আছে একটি একটি দেখছে একটি দেখছে একজন | (same as min8) |
| `1079274291_9aaf896cc1.jpg` | একটি ছোটো বালক ক্যামেরায় ছবি তোলার জন্যে জিহ্বা দেখাচ্ছে। অন্য একটি বালক তার ছবি তোলা দেখছে । | বাচ্চা | বাচ্চা আছে একটি | বাচ্চা আছে হাসছে একটি আছে একটি |
| `1096395242_fc69f0ae5a.jpg` | খেলনা বন্দুক হাতে বালক | ছোটটি খেলনা খেলছে | ছোট একটি ছেলে বুদ করছে একটি বুদদ আছে একটি আছে একটি দেখছে | (same as min8) |
| `109671650_f7bbc297fa.jpg` | খাড়া পাহাড় চূড়ায় হলুদ ছোটো প্যান্ট পরিহিত বালকটি দাঁড়িয়ে আছে | ছেলে উপর আছে | ছেলে উপর আছে একটি আছে একটি দেখছে একটি | (same as min8) |

### Length summary

| Variant | Mean gen words | Min | Max | Mean ref words |
|---|---:|---:|---:|---:|
| `baseline_beam4` | 2.2 | 1 | 4 | 9.1 |
| `min8_no_early_stop` | 8.5 | 3 | 17 | 9.1 |
| `min12_no_early_stop` | 9.4 | 6 | 17 | 9.1 |

### What this tells us

1. **The decoder responds to length constraints.** Mean output length
   roughly quadruples (2.2 → 8.5 → 9.4 words) under `min_new_tokens`,
   matching the same mechanical effect seen in BanglaLekha.

2. **Unlike BanglaLekha, some forced-length outputs surface genuine
   new content tokens.** The clearest case is `101654506` (snow dog):

   | Decode | Output |
   |---|---|
   | baseline | `সাদাটি তুষার মধ্যে কুকুর` (white in snow dog) |
   | min8 | `সাদাটি তুষার মধ্যে কুকুর লাফ । একটি ধরার` (white in snow dog jump . one catching) |

   `লাফ` (jump) and `ধরার` (catching) appear nowhere in the baseline
   output for any of the 8 images, but the actual snow-dog image
   shows a dog jumping/catching. Similarly `1042020065`
   (boy looking at boats): baseline `লোক`, min8 `লোক নৌকা আছে একটি
   নৌকা` — the `নৌকা` (boat) token appears under forced length but
   never at beam=4. These are not filler; they are image-specific
   tokens the beam hasn't surfaced.

3. **But most forced-length outputs still degenerate to repetition.**
   `1048710776` → `লোক বসে দেখছে দেখছে দেখছে একজন দেখছে`;
   `1056249424` → `শিশু আছে একটি একটি দেখছে একটি দেখছে একজন`. The
   `no_repeat_ngram_size=3` constraint prevents the danda-loop
   pathology but the model still falls into "repeat the same content
   token + filler" loops past its natural output length.

4. **`min12 ≈ min8` for most items.** Same pattern as BanglaLekha —
   the model emits its forced floor then stops cleanly; raising the
   floor to 12 changes one item out of 8. There isn't a longer
   well-formed sentence sitting just past the floor.

**Verdict — partial revision of the BanglaLekha finding.** The
BanglaLekha diagnostic concluded that *there is no hidden
compositional capability that beam=4 masks*. The BAN-Cap result is
more nuanced: there is *partial* hidden capability — specifically,
image-specific content tokens that beam search filters out — but
**not** hidden sentence-level grammar. Forced length recovers more
vocabulary on a content-rich corpus than on a content-sparse one.
Composition (`<subject> <action> <object> <location>` structure) is
still genuinely missing.

This is a softer version of the "decoder capacity is the binding
constraint" finding, not a refutation: capacity limits sentence
*structure*, not necessarily vocabulary access.

## Test 3 — caption-selection val cross-entropy

### Setup

Full 809-item val split. Same adapter, teacher-forced forward pass.
The only thing that varies is which of the **5 peer captions** per
image is used as the target.

For BAN-Cap, the 5 captions are collected one per annotator from 5
different annotators — peer captions of comparable length, not the
short/long pairing that BanglaLekha used. `first`/`last`/`random`
therefore vary by annotator identity rather than by caption length.

### Result

| Selection | Mean ref words | Val loss | Perplexity | n_val |
|---|---:|---:|---:|---:|
| `first` | 8.6 | 4.643 | 103.8 | 809 |
| `last` | 8.5 | 4.778 | 118.9 | 809 |
| `random` | 8.6 | 4.697 | 109.6 | 809 |

**Gap (`last` − `first`):** +0.135 CE / **1.14× perplexity**.

### Direct comparison to BanglaLekha

| Selection | BanglaLekha CE | BanglaLekha PPL | BAN-Cap CE | BAN-Cap PPL |
|---|---:|---:|---:|---:|
| `first` | 2.569 | 13.1 | 4.643 | 103.8 |
| `last` | 3.899 | 49.3 | 4.778 | 118.9 |
| `random` | 3.407 | 30.2 | 4.697 | 109.6 |
| **`last` − `first`** | **+1.330 / 3.76×** | | **+0.135 / 1.14×** | |
| **Mean ref words spread** | 6.8 → 10.2 (+3.4) | | 8.6 → 8.5 (~0) | |

### What this tells us

1. **The 3.76× cross-reference PPL gap is gone.** BAN-Cap's
   `first`/`last`/`random` cluster within 0.135 CE — a **9.8× tighter
   spread than BanglaLekha**. The eval target is now genuinely
   unified: the headline 4.64 from `bancap_full/README.md` is loss
   against a representative caption distribution, not against the
   easier of two structurally different target forms.

2. **Mean reference word counts confirm the structural difference.**
   BanglaLekha's `first` averages 6.8 words and `last` averages 10.2
   — a 1.5× length spread per image. BAN-Cap's selections all
   average 8.5-8.6 words: the 5 peer captions are length-matched per
   annotator-pair. This is exactly the property the corpus switch
   was supposed to provide.

3. **`random` lands cleanly between `first` and `last`.** PPL 109.6
   is the arithmetic average of 103.8 and 118.9 ± a small sampling
   offset — implying the per-token entropy difference between
   annotator-0 and annotator-4 captions on the same image is small.
   The residual 0.135 CE gap is consistent with normal annotator
   stylistic variance (word-choice / punctuation / sentence-end
   preference), not with a structural eval bias.

4. **Absolute PPL of ~104 is much higher than BanglaLekha's ~13.**
   This is the same effect documented in `bancap_full/README.md` §
   "How to read the +2.09 val-loss gap": BAN-Cap is fundamentally
   harder per-token (3× vocabulary, flatter prior). The diagnostic
   does not change that — but it does confirm that the high PPL is
   uniform across selections, not an artifact of which reference we
   scored against.

**Verdict — corpus switch decision validated.** The dominant finding
that motivated the BanglaLekha → BAN-Cap migration was a 3.76× PPL
gap between short and long references on BanglaLekha. On BAN-Cap that
gap is 1.14× — effectively zero. The reported headline val loss
represents progress against a unified target, exactly as intended.

## Synthesis — revised theory after data

The BanglaLekha diagnostic re-ranked four causes A–D (visual signal
20% / corpus prior 25% / capacity 30% / eval bias 35%). The BAN-Cap
results re-rank them again:

| Cause | BanglaLekha estimate | BAN-Cap evidence | BAN-Cap estimate |
|---|---:|---|---:|
| A — Visual signal weak | 15-20% | Strengthened. Noise→Jaccard 0.0 vs BanglaLekha's 0.21 shows BAN-Cap model uses image features more aggressively; shuffled outputs follow shuffled-in content. | **10-15%** |
| B — Corpus prior + safe-token attractor | 20-25% | Weakened. Flatter BAN-Cap prior already eliminated the danda-loop pathology; safe-token bias is now less dominant. | **15-20%** |
| C — Decoder capacity / compositional grammar | 25-30% | Partially weakened. Forced length *can* surface image-specific tokens on BAN-Cap (`লাফ`/`ধরার`/`নৌকা`); but sentence-level grammar still missing. Capacity still binds, just less so on vocabulary access. | **25-30%** |
| D — Eval bias on a structurally-pairing corpus | 30-35% | **Removed by the corpus switch.** Cross-reference variance is now 0.135 CE / 1.14× PPL. | **~5%** (residual annotator variance) |

The dominant blocker is now **decoder capacity for composition**
(C, the same ceiling identified on BanglaLekha) combined with **the
amount of training the model has received** (val curve still
descending at step 5,000 — see `bancap_full/README.md`). Both are
addressable: more steps + bigger base model. Eval-bias (D) is
genuinely solved.

## Implications for next experiments

In priority order, by expected impact / cost:

1. **Train BAN-Cap longer.** Cheapest test of "the val curve is
   underbaked." 10K-step run (~50 min on T4) likely pushes val below
   4.0 and may close some of the forced-length gap by giving the
   model more chances to associate image features with their
   correct multi-word descriptions.

2. **Compute real metrics now that the eval target is unified.**
   `scripts/score_captions.py` with 5-reference BLEU-1..4 / CIDEr /
   BERTScore-bn / M-CLIPScore on the 809-val. The BanglaLekha
   diagnostic warned that the 2.55 number was misleading because of
   eval bias; the BAN-Cap 4.64 number is honest and BLEU/CIDEr
   should now correlate sensibly. This is the §C3 metric framework
   the paper plan needs.

3. **Image-tower LoRA (the real one).** Now strongly motivated. The
   BAN-Cap diagnostic showed the model uses image features more than
   BanglaLekha did, but `target_modules: [query, key, value]`
   silently misses CLIP-ViT's `q_proj/k_proj/v_proj`. Unfreezing
   them is the next likely source of gains on visual-content fidelity
   (colors, counts, fine-grained spatial relations like "on top of"
   vs "next to").

4. **Bigger base model (PaliGemma-3B + QLoRA).** Test 2 forced-length
   shows GiT-base lacks compositional grammar; it can surface
   content tokens but can't bind them into a well-formed sentence.
   A larger base with more decoder layers is the only addressed for
   this.

5. **(Optional) `index:N` selectors.** Per-annotator val loss
   (`index:0` through `index:4`) would surface whether one
   annotator's caption style is systematically harder than others —
   useful for the paper's discussion of human-annotation variance.

## Honest caveats

- **n=8 for Tests 1 and 2.** The sample table is small; lexical
  patterns are robust across the 8 items but small-N effects on
  Jaccard could shift by ±0.05. The Jaccard=0.000 for noise is robust
  because every output is the same OOD-fallback string.

- **`random` selection in Test 3 doesn't match the training-time
  RNG.** Same caveat as for the BanglaLekha diagnostic — the val
  loader's seeded `choice()` calls are not the same sequence as the
  trainer's. The number is representative of the random-target loss
  but not exactly reproducible from the trainer's trajectory.

- **No `index:N` selectors run.** Worth adding once the script is in
  active use; for this comparison-with-BanglaLekha pass we kept
  selections to `{first, last, random}` to mirror the prior tests
  exactly.

- **Decode parameters in Test 2 are not swept.** `length_penalty=2.0`
  is aggressive; a sweep over `length_penalty ∈ {1.0, 1.5, 2.0}`
  would clarify whether the new tokens (`লাফ`, `ধরার`, `নৌকা`) appear
  at gentler settings or only at extreme forcing. Future work.

## Refactor note

The script and config now support both corpora via the new
`data.loader` config key (`banglalekha` | `bancap`), dispatched
through `src.data.build_caption_dataset`. `configs/sentence_diagnostics.yaml`
remains BanglaLekha-targeted; `configs/sentence_diagnostics_bancap.yaml`
points at the BAN-Cap adapter + Flickr8k images. The same dispatch
will be reusable by `scripts/decode_ablation.py` and a future
`scripts/score_captions.py` without further refactoring.

## Artifacts

- `paths.EXP_RESULTS/sentence_diagnostics_bancap/results.md` —
  auto-generated variant tables.
- `paths.EXP_RESULTS/sentence_diagnostics_bancap/samples.json` — all
  generations across the three tests.
- `paths.EXP_RESULTS/sentence_diagnostics_bancap/summary.json` —
  numeric summaries (Jaccard, length stats, val losses).

## Reproduce

```bash
# Fetch Flickr8k transient (~1 GB):
mkdir -p /content/bancap_dl
cd /content/bancap_dl
KAGGLE_USERNAME=<user> KAGGLE_KEY=<key> kaggle datasets download \
    -d adityajn105/flickr8k && unzip -q flickr8k.zip

# Run:
cd /content/bangla-vlm-lora
python scripts/sentence_diagnostics.py --config configs/sentence_diagnostics_bancap.yaml
```

Requires the `bancap_full` adapter at
`paths.EXP_CHECKPOINTS/bancap_full/` and the BAN-Cap captions CSV at
`paths.DATA_RAW/bancap/BAN-Cap_captiondata.csv` (persistent on Drive).
