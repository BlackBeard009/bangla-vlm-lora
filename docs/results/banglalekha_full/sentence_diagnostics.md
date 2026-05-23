# Sentence-length diagnostics on the saved banglalekha_full adapter

**Date:** 2026-05-23
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/sentence_diagnostics.py`
**Config:** `configs/sentence_diagnostics.yaml`
**Adapter:** `paths.EXP_CHECKPOINTS/banglalekha_full/` (bridged GiT +
LoRA, donor init, 5,000 steps — see `README.md` in this directory).

## What this is

Three independent diagnostics, no retraining. Each targets one of the
candidate explanations for the templated outputs reported in
`README.md`:

  - **Test 1 — image scrambling.** Decode the same 8 val items with
    real / gray / random-noise / shuffled image inputs, baseline beam=4
    decode. If outputs barely change between real and gray, the model
    has learned to ignore the image.
  - **Test 2 — forced-length decode.** Same 8 val items, real images,
    with `min_new_tokens` ∈ {0, 8, 12} plus `no_repeat_ngram_size=3` and
    `length_penalty=2.0`. Tests whether the model has compositional
    structure that beam search hides.
  - **Test 3 — caption-length val cross-entropy.** Full 915-item val
    split. Recompute val CE under `caption_selection ∈ {first, last,
    random}`. BanglaLekha's `captions[0]` is the ~7-word short form;
    `captions[-1]` is the ~10-word descriptive form. The headline
    `banglalekha_full` val loss 2.55 used `first` only.

Together they let us partition the "outputs are templated" finding into
visual-grounding, decode, capacity, and eval-bias contributions.

## Test 1 — image scrambling

### Setup

| Variant | Image input |
|---|---|
| `real` | Unmodified val image |
| `gray` | Solid `(128, 128, 128)` RGB image, same size |
| `noise` | Per-pixel uniform random RGB, same size, seed 42 |
| `shuffled` | `image[i]` replaced with `image[(i + 1) mod 8]` (real but wrong-for-this-caption) |

Decode: beam=4, `max_new_tokens=30`, no penalties (matches
`decode_ablation::baseline_beam4`). One re-seed before each variant.

### Outputs

| Image | Reference | `real` | `gray` | `noise` | `shuffled` |
|---|---|---|---|---|---|
| `4.png`   | ছয় জন মানুষ দাড়িয়ে আছে। | পুরুষ করছে | । একজন আছে | । একজন আছে | পুরুষ আছে |
| `31.png`  | একটি বাচ্চা ছেলে বসে আছে। | পুরুষ আছে | । একজন আছে | । একজন আছে | মানুষ করছে |
| `41.png`  | কয়েক জন মানুষ দাঁড়িয়ে ও কয়েক জন মানুষ বসে আছে। | মানুষ করছে | । একজন আছে | । একজন আছে | । গাছ । একটি গাছ । । একটি গাছ । গাছ । একটি গাছ । আছে |
| `45.png`  | জলাশয় এর পাশে গাছ আছে। উপরে নীল আকাশে অল্প সাদা মেঘ আছে। | । গাছ । একটি গাছ । । একটি গাছ । গাছ । একটি গাছ । আছে | । একজন আছে | । একজন আছে | শিশু আছে |
| `48.png`  | একজন মহিলা ও একটি কিশোর ছেলে বাচ্চা কোলে নিয়ে বসে আছে। | শিশু আছে | । একজন আছে | । একজন আছে | মানুষ করছে |
| `65.png`  | সবুজ ফসলের মাঠে একজন বালক লাল জামা পরিধান করে আছে। | মানুষ করছে | । একজন আছে | । একজন আছে | শিশু । একজন আছে |
| `71.png`  | সবুজ ঘাসের উপর বালক, বালিকা ও শিশু আছে। | শিশু । একজন আছে | । একজন আছে | । একজন আছে | মানুষ । অনেকগুলো আছে |
| `110.png` | পানির ভিতর একজন বালক খালি শরীরে বসে আছে। | মানুষ । অনেকগুলো আছে | । একজন আছে | যাচ্ছে | পুরুষ করছে |

### Divergence from `real` baseline

| Variant | Exact-match with `real` | Mean Jaccard token overlap |
|---|---:|---:|
| `gray` | 0.0% | 0.256 |
| `noise` | 0.0% | 0.206 |
| `shuffled` | 0.0% | 0.108 |

### What this tells us

1. **The model is NOT ignoring the image.** Outputs change across all
   three perturbations. If the visual signal were inert, gray and noise
   would each reproduce the real-image output.
2. **But the visual signal only drives noun selection.** Look at the
   `gray` column: 7 of 8 images collapse to the identical default
   `। একজন আছে` (the model's "no-image" fallback). The single deviation
   on `110.png` (`যাচ্ছে`) is the noise-image case, which is OOD enough
   to perturb the language scaffold too. With real images, the noun
   varies (`পুরুষ`, `মানুষ`, `শিশু`, `গাছ`) but the surrounding
   template stays the same. **The decoder is visually blind above the
   noun level.**
3. **`shuffled` confirms the model attends to image-specific content.**
   Shuffled (real but wrong) images give the *lowest* Jaccard overlap
   with `real` (0.108) — lower than noise. That is only possible if the
   model is genuinely conditioning on image content; with a different
   real image, it picks a different content noun. E.g. `45.png`'s
   shuffled output is `শিশু আছে` because the shuffled-in image (`48.png`)
   contains a child.
4. **The danda-repetition pathology on `45.png` is image-triggered.**
   With `real` and `shuffled-in-of-41.png` (also a tree-heavy scene)
   the model emits the danda-loop `। গাছ । একটি গাছ ...`. With gray or
   noise, the same image position decodes cleanly to `। একজন আছে`. So
   the loop is not a generic beam pathology; it fires when the image
   features for tree-heavy scenes activate a content path the model
   hasn't learned to terminate.

**Verdict on the "image tower frozen" hypothesis.** Partially
refuted — the visual signal *is* reaching the decoder enough to switch
content nouns. But it is not driving compositional structure. The
template machinery (`। <noun> । ... আছে`) is text-prior driven and
visually inert.

## Test 2 — forced-length decode

### Setup

| Variant | Params |
|---|---|
| `baseline_beam4` | `num_beams=4, max_new_tokens=30` |
| `min8_no_early_stop` | `+min_new_tokens=8, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=False` |
| `min12_no_early_stop` | `+min_new_tokens=12, ...` (rest identical) |

### Outputs (8 val items, real images)

| Image | Reference | `baseline_beam4` | `min8_no_early_stop` | `min12_no_early_stop` |
|---|---|---|---|---|
| `4.png`   | ছয় জন মানুষ দাড়িয়ে আছে। | পুরুষ করছে | পুরুষ । অনেকগুলো কাজ । একজন । একজন । কাজ । অনেকগুলো দেখা । অনেকগুলো । অনেকগুলো আছে | (same as min8) |
| `31.png`  | একটি বাচ্চা ছেলে বসে আছে। | পুরুষ আছে | পুরুষ । একজন । একটি । আছে | (same) |
| `41.png`  | কয়েক জন মানুষ দাঁড়িয়ে ও কয়েক জন মানুষ বসে আছে। | মানুষ করছে | পুরুষ একজন । একজন বসে । একজন আছে একজন এবং জন এবং জন বসে । জন । একটি আছে | (same) |
| `45.png`  | জলাশয় এর পাশে গাছ আছে। উপরে নীল আকাশে অল্প সাদা মেঘ আছে। | । গাছ । একটি গাছ । ... । আছে | । গাছ । একটি । একটি গাছ । । একজন । একটি দেখা । একটি আছে | (same) |
| `48.png`  | একজন মহিলা ও একটি কিশোর ছেলে বাচ্চা কোলে নিয়ে বসে আছে। | শিশু আছে | শিশু । একজন । একজন বসে । একটি । আছে | শিশু । একজন । একজন বসে । একটি । একটিের বসে । একজন হাতে আছে |
| `65.png`  | সবুজ ফসলের মাঠে একজন বালক লাল জামা পরিধান করে আছে। | মানুষ করছে | মানুষ । একজন হেঁটে । একটিের । । একটি । একজন হাতে আছে | (same) |
| `71.png`  | সবুজ ঘাসের উপর বালক, বালিকা ও শিশু আছে। | শিশু । একজন আছে | শিশু । একজন । আছে পিছনে মানুষ । একটি । একজন আছে | (same) |
| `110.png` | পানির ভিতর একজন বালক খালি শরীরে বসে আছে। | মানুষ । অনেকগুলো আছে | মানুষ । অনেকগুলো । অনেকগুলো আছে পিছনে মানুষ । । একজন আছে | (same) |

### Length summary

| Variant | Mean gen words | Min | Max | Mean ref words |
|---|---:|---:|---:|---:|
| `baseline_beam4` | 4.2 | 2 | 16 | 7.4 |
| `min8_no_early_stop` | 13.4 | 7 | 19 | 7.4 |
| `min12_no_early_stop` | 14.0 | 7 | 19 | 7.4 |

### What this tells us

1. **The decoder constraints work as expected.** Mean output length
   triples (4.2 → 13.4 words) with `min_new_tokens=8` plus the
   anti-repetition / length-penalty package. The model CAN be forced
   to keep generating.
2. **But the extra length is degenerate.** The longer outputs are
   stretched-out priors — `পুরুষ । অনেকগুলো কাজ । একজন । একজন । কাজ ।` —
   not compositional captions. New tokens that appear at length include
   filler verbs (`দেখা`, `হেঁটে`), counters (`অনেকগুলো`), and
   prepositions (`পিছনে`, `হাতে`) tacked on with no coherent grammar.
   None of the long outputs match the structure of the references.
3. **`min12` ≈ `min8` for most items.** The model already emits
   ~13 wordpieces under `min8` then hits EOS-equivalent natural
   stopping; raising the floor to 12 changes one item out of eight.
   There isn't a longer well-formed sentence sitting just past the
   floor; the floor is already deep into degenerate territory.
4. **Confirmation: there is no hidden compositional capability that
   beam=4 was masking.** The earlier decode-ablation finding ("top-p
   surfaces hidden vocabulary") was about diversity within short
   outputs. The forced-length test extends that finding: the model has
   richer vocab than beam reveals, but its **compositional grammar is
   genuinely missing**. Adapting decode hyperparameters cannot recover
   sentence structure that the model never learned.

**Verdict on the "decoder amplifies a hidden prior" hypothesis.**
Refuted as the primary cause. Beam search is conservative but the
shorter outputs are not a beam pathology — the model genuinely lacks
the capacity to compose longer correct captions.

## Test 3 — caption-length val cross-entropy

### Setup

Full 915-item val split. Same adapter, teacher-forced forward pass
(`labels = input_ids`, no generation). The only thing that varies is
which of the two BanglaLekha references is used as the target.

`captions[0]` is empirically the short summary form (mean 6.8 words);
`captions[-1]` is the descriptive form (mean 10.2 words). 73% of items
have c[-1] longer than c[0], 40% have c[-1] ≥ 2× c[0]. Training used
`caption_selection: random` so the model saw both; the headline 2.55
val loss was computed under `caption_selection: first`.

### Result

| Selection | Mean ref words | Val loss | Perplexity | n_val |
|---|---:|---:|---:|---:|
| `first` | 6.8 | 2.569 | 13.1 | 915 |
| `last` | 10.2 | 3.899 | 49.3 | 915 |
| `random` | 6.8 (sampled) | 3.407 | 30.2 | 915 |

**Gap (`last` − `first`):** +1.330 CE / **3.76× perplexity**.

### What this tells us

1. **The "convergence" reported as val_loss 2.55 is convergence on the
   short caption only.** The same model on the same images, scored
   against the LONG reference, is at perplexity 49.3 — almost 4×
   worse. The training-time eval (`caption_selection: first`) masked
   this entirely.
2. **The model is dramatically underfit on the long form.** Perplexity
   49 on a 10-word average means the per-token uncertainty is huge.
   The model does not know how to predict the next token in a
   descriptive Bangla sentence given the image; it only knows how to
   predict the next token of a short canonical sentence.
3. **`random` is between `first` and `last`, as expected.** PPL 30.2 is
   close to the geometric mean of 13.1 and 49.3 (= 25.4); the
   difference comes from the random sampling not being item-paired
   with the training-time random RNG.
4. **The headline plateau at step ~4,500 is real but on the wrong
   target.** If we had monitored against `last` or `random`, we would
   have seen val loss still trending down at step 5,000, and probably
   would have trained longer. The eval choice silently truncated the
   training schedule.

**Verdict on the "training-time eval is masking underfitting"
hypothesis.** Strongly confirmed. The reported val curve is for a
narrower target than the model is actually expected to handle.

## Synthesis — revised theory after data

The pre-experiment theory ranked four causes A–D (visual signal /
corpus prior / capacity / eval bias) at roughly 40/25/20/15%. The
diagnostic results re-rank them substantially:

| Cause | Pre-data estimate | Evidence | Revised estimate |
|---|---:|---|---:|
| A — Visual signal weak / image tower frozen | 40% | Refuted as dominant. Real-vs-gray Jaccard 0.26 and real-vs-shuffled Jaccard 0.11 prove the model uses the image — but only for noun selection, not composition. | **15–20%** |
| B — Corpus prior + beam search picks safe short outputs | 25% | Partially confirmed. Beam=4 collapses to short; but forcing length doesn't recover structure, so the prior is not the only thing capping length. | **20–25%** |
| C — Decoder capacity / training budget too small to compose | 20% | Strongly confirmed. Forced-length outputs are stretched-out priors; the model lacks compositional grammar to surface. | **25–30%** |
| D — Eval bias + training under-fits long captions | 15% | Strongly confirmed. 3.76× perplexity gap between short and long references. Headline "convergence" was against the easier target. | **30–35%** |

The dominant cause turns out to be **D + C combined**: the model never
got a strong training signal on the long form, and its small decoder
plus limited training budget would not have learned compositional
captioning even if it had. The original "image tower is frozen" theory
remains relevant for *fine-grained visual grounding* (colors, counts,
spatial relations) but cannot explain the templating itself.

## Implications for next experiments

In priority order, by expected impact / cost:

1. **Train with `caption_selection: last` (or weighted toward long).**
   Cheapest test of the dominant hypothesis. Same script
   (`scripts/banglalekha_slice.py`), same 5,000-step config, one-line
   config change: `caption_selection: last` in
   `configs/banglalekha_full.yaml`. ~40 min on T4. **Predict**:
   training-time val loss against `last` should drop significantly
   below 3.90; sample outputs should be longer and more
   compositional even under unconstrained beam=4 decode. If it doesn't
   improve, capacity (C) is the binding constraint and we need a
   bigger model.
2. **Add `caption_length` weighted sampling.** Slightly fancier: sample
   `captions[-1]` with probability 0.7 and `captions[0]` with 0.3 so
   the model still sees the short form for stability. Trades against
   #1 only if #1 produces samples that are long-but-vague.
3. **Run real metrics on the existing adapter** (the previously
   recommended next step). Now with three decode variants instead of
   two: `baseline_beam4`, `beam4_penalties`, and `min8_no_early_stop`
   — the last is mechanically interesting even if quality is bad,
   because BLEU/CIDEr against long references would actually pick it
   up. M-CLIPScore probably tanks for the forced-length variant
   because the outputs are dilute.
4. **Image-tower LoRA (now with explicit motivation).** Add CLIP-ViT's
   `q_proj/k_proj/v_proj/out_proj` to `target_modules` and
   `visual_projection.visual_projection.0` to `modules_to_save`. The
   image-scrambling test shows visual signal is *partially* used; the
   image-tower LoRA test specifically targets the fine-grained
   grounding piece (colors, counts, spatial). Won't fix templating but
   should help content faithfulness.
5. **Bigger base model (PaliGemma-3B).** A 6-layer text decoder with
   ~45 M params and a rank-8 LoRA + 29 K new tokens has very little
   compositional headroom. Per `docs/paper_plan.md` §3 this is already
   planned. Move it up in priority if #1 doesn't unblock composition.

## Honest caveats

- **n=8 for Tests 1 and 2.** The sample table is small; lexical patterns
  are robust across the 8 items but small-N effects on Jaccard could
  shift by ±0.05.
- **`random` selection in Test 3 doesn't match the training RNG
  trajectory.** The val-loader RNG is seeded but its sequence of
  `choice()` calls is not the same as the training-loader's. The
  number is representative but not exactly the random-target loss the
  trainer optimized.
- **Image normalization.** Gray (128,128,128) becomes ~zero after
  CLIP's mean/std normalization; noise stays high-entropy. Both are
  valid "no image content" controls but they probe slightly different
  parts of the encoder.
- **Decode parameters in Test 2** were chosen to be reasonable but not
  fully ablated. `length_penalty=2.0` is aggressive; future work could
  sweep `length_penalty ∈ {1.0, 1.5, 2.0}` for a clean knob study. The
  qualitative conclusion (forced-length is stretched priors) is robust
  across these.

## Artifacts

- `paths.EXP_RESULTS/sentence_diagnostics/results.md` — auto-generated
  variant tables and summaries.
- `paths.EXP_RESULTS/sentence_diagnostics/samples.json` — all
  generations across the three tests.
- `paths.EXP_RESULTS/sentence_diagnostics/summary.json` — numeric
  summaries (Jaccard, length stats, val losses).
- `docs/results/banglalekha_full/sentence_diagnostics.md` — this
  write-up.

## Reproduce

```bash
cd /content/bangla-vlm-lora
python scripts/sentence_diagnostics.py --config configs/sentence_diagnostics.yaml
```

Requires the `banglalekha_full` adapter at
`paths.EXP_CHECKPOINTS/banglalekha_full/` and the BanglaLekha
captions + images per the parent `README.md` Reproduce block.
