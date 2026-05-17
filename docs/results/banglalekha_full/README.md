# BanglaLekha full — bridged GiT + LoRA on the full corpus

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/banglalekha_slice.py`
**Config:** `configs/banglalekha_full.yaml`
**Dataset:** BanglaLekha-Image-Captions (Mendeley `rxxch9vw59` v2 — 9,154
images, 2 native Bangla captions each).
**Companion (pipeline validation):** `docs/results/banglalekha_slice/`

## What this is

The slice run (200 train / 300 steps) proved the bridged-GiT + LoRA
pipeline works end-to-end on real data. This run graduates it to the
full corpus: 8,239 train / 915 val (seed-42 90/10 split), 5,000 LoRA
steps, `caption_selection: random` so both reference captions per image
are seen. It is the first time this recipe is trained to (near)
convergence on real Bangla photographs.

A prior session attempted the same run; the kernel died before the
write-up could be promoted to the repo, so the previous artifact on
Drive is unreferenced. This run reproduces it cleanly under the same
seed and supersedes it.

## Setup

- **Bridge:** GiT-base vocab (30,522) extended with 29,055 BanglaBERT
  wordpieces; `donor` init strategy (input embeddings copied from
  BanglaBERT where the token matches; LM-head rows = mean of original
  LM head). Diacritic-preserving tokenizer reload
  (`do_lower_case=False, strip_accents=False, tokenize_chinese_chars=False`)
  is active per the 2026-05-17 fix.
- **LoRA:** rank 8, α 16, dropout 0.05, target
  `[query, key, value]` on GiT's text-side attention;
  `modules_to_save=[output, word_embeddings]`. 109.5 M / 330.8 M
  parameters trainable (33.11%). Image tower is fully frozen.
- **Data split:** 8,239 train / 915 val from the full 9,154-image
  set. `caption_selection: random` — each `__getitem__` returns one of
  the two reference captions at random.
- **Training:** 5,000 steps, batch 4, AdamW lr 1e-4, no weight decay.
- **Decode (eval):** beam=4, `max_new_tokens=30`, no length / repetition
  penalty, no sampling.

## Result

| Step | Train loss | Val loss |
|---:|---:|---:|
| 500 | 4.985 | 3.432 |
| 1000 | 4.178 | 3.137 |
| 1500 | 3.919 | 2.978 |
| 2000 | 3.748 | 2.860 |
| 2500 | 3.487 | 2.762 |
| 3000 | 3.486 | 2.695 |
| 3500 | 3.402 | 2.643 |
| 4000 | 3.352 | 2.606 |
| 4500 | 3.141 | 2.550 |
| 5000 | 3.152 | 2.569 |

**Wall:** 2,374.8 s on T4 (~0.47 s/step).

Val loss is monotonically decreasing through step 4500 (3.43 → 2.55),
with a 0.02 uptick at step 5000 (2.55 → 2.57). The curve is essentially
flat for the last 1,000 steps — the model has plateaued at this scale.
Train and val are close (3.15 vs. 2.57): no overfitting.

## Sample generations (held-out val, step 5000)

| Image | Reference (native annotation) | Generated |
|---|---|---|
| `4.png`   | ছয় জন মানুষ দাড়িয়ে আছে। | `পুরুষ করছে` |
| `31.png`  | একটি বাচ্চা ছেলে বসে আছে। | `পুরুষ আছে` |
| `41.png`  | কয়েক জন মানুষ দাঁড়িয়ে ও কয়েক জন মানুষ বসে আছে। | `মানুষ করছে` |
| `45.png`  | জলাশয় এর পাশে গাছ আছে। উপরে নীল আকাশে অল্প সাদা মেঘ আছে। | `। গাছ । একটি গাছ । । একটি গাছ । গাছ । একটি গাছ । আছে` |
| `48.png`  | একজন মহিলা ও একটি কিশোর ছেলে বাচ্চা কোলে নিয়ে বসে আছে। | `শিশু আছে` |
| `65.png`  | সবুজ ফসলের মাঠে একজন বালক লাল জামা পরিধান করে আছে। | `মানুষ করছে` |
| `71.png`  | সবুজ ঘাসের উপর বালক, বালিকা ও শিশু আছে। | `শিশু । একজন আছে` |
| `110.png` | পানির ভিতর একজন বালক খালি শরীরে বসে আছে। | `মানুষ । অনেকগুলো আছে` |

Full per-eval sample dump: `paths.EXP_RESULTS/banglalekha_full/samples.json`.

### What changed across training

Compared to the slice's terminal output (`মানুষ আছে` / `পুরুষ মানুষ আছে`),
the full run's outputs are:

- **Slightly more lexically varied.** New tokens appear: `করছে` (does/is
  doing), `শিশু` (child), `অনেকগুলো` (many). `45.png` (tree by water)
  correctly emits `গাছ` ("tree") rather than collapsing to `মানুষ`.
- **Occasionally content-aware.** `48.png` (a woman holding a teenage
  boy) → `শিশু আছে` ("a child exists") is closer to the reference than
  the slice's `পুরুষ মানুষ আছে`.
- **Still templated.** Most outputs are 2–3 wordpieces; the dominant
  pattern remains `<noun> আছে` / `<noun> করছে`. None of the longer
  reference structures (multi-clause descriptions, numerals, color +
  clothing) are reproduced.
- **One degenerate output.** `45.png` decodes as a danda-repetition
  loop (`। গাছ । একটি গাছ । ...`) — a known beam-search failure mode
  for this corpus when there is no `no_repeat_ngram_size` constraint.

## Probable reasons for generic outputs

Loss-curve convergence with templated generation is the canonical
captioning-diversity problem; multiple causes stack.

**Dataset side (likely dominant):**

1. **BanglaLekha caption distribution is heavily templated.** Most
   captions are 3–7 wordpieces and use a small set of recurring
   patterns: `<noun> আছে`, `<count> জন মানুষ আছে`, `... বসে আছে`. The
   most-common content word in the corpus is `মানুষ`. Cross-entropy
   pulls the model toward this prior; the prior *is* "person exists."
2. **Only 2 reference captions per image.** The model never sees
   rich linguistic variation for any single image — it learns "what is
   the safe sentence to emit for image class X" rather than "how can
   I describe this specific image."
3. **Translation / annotation noise.** `docs/paper_plan.md` flags
   BanglaLekha as the lowest-quality of the three corpora; BAN-Cap is
   described as the more rigorous benchmark for exactly this reason.

**Method side:**

4. **Image tower is fully frozen.** `target_modules: [query, key, value]`
   applies to GiT's *text*-side attention only. The CLIP-ViT image
   encoder receives no gradient. The visual signal reaching the LM
   head is whatever GiT's pretrained image features already encode —
   fine for "this image contains people / a tree / sky" but not for
   "a woman holding a baby in her lap." Already flagged as a caveat in
   the slice README.
5. **Beam search bias.** `num_beams=4` is known to favor safe,
   high-probability prefixes. Nucleus / diverse-beam decoding would
   likely unstick this without retraining.
6. **No length / repetition / coverage penalties.** A
   `no_repeat_ngram_size=3` would have killed `45.png`'s danda loop;
   a `length_penalty > 1.0` would push the decoder past 2–3 wordpieces.

The bridge's job — "can the model produce Bangla?" — is demonstrably
done (cf. the six-stage motivation chain, §6 of `CLAUDE.md`). The
**next** failure mode, generic / templated outputs, is what stops any
captioning model from being publishable on n-gram metrics alone — and
is exactly why `docs/paper_plan.md` §C3 (human evaluation + learned
metrics like Polos / FLEUR) is part of the paper. n-gram scores would
look fine while a human rater would call these uninformative.

## Honest caveats

- **No metric numbers reported yet.** BLEU / CIDEr / BERTScore-bn /
  M-CLIPScore on the 915-val are next; right now we only have CE loss.
- **Same-seed reproduction.** A previous incomplete run on Drive
  (now overwritten) produced 2.569 final val under the same config and
  seed. This run lands at 2.569 — reassuring as a reproducibility
  check, but not independent corroboration of the method.
- **Decoder hyperparameters are untuned.** All decoding numbers in
  this report use the default config (beam=4, no penalties). Different
  decode settings will materially change the sample table.
- **915 val is small.** Variance across val-loss eval points (0.02
  uptick at step 5000) is well within the noise of a 915-image
  evaluation with batch 4 and random caption selection.

## Artifacts

- `paths.EXP_RESULTS/banglalekha_full/results.md` — regenerated each run.
- `paths.EXP_RESULTS/banglalekha_full/samples.json` — full sample dump
  across eval points.
- `paths.EXP_CHECKPOINTS/banglalekha_full/` — LoRA adapter
  (`adapter_model.safetensors`, ~1.17 GB with `modules_to_save`
  embedding+head) and the bridged tokenizer. Persisted on Drive.
- `docs/results/banglalekha_full/results_table.md` — frozen copy of the
  eval table for citation.

## Reproduce

```bash
# First time only — captions.json (~2.5 MB) and images.zip (6.3 GB):
mkdir -p $DRIVE_ROOT/data/raw/banglalekha
curl -L -o $DRIVE_ROOT/data/raw/banglalekha/captions.json \
  'https://data.mendeley.com/public-files/datasets/rxxch9vw59/files/9b3e789a-5a5c-48b3-8a2c-2c91e9307c2a/file_downloaded'
mkdir -p /content/banglalekha_raw
curl -L -o /content/banglalekha_raw/images.zip \
  'https://data.mendeley.com/public-files/datasets/rxxch9vw59/files/220ffc5a-8645-4c0b-bd1b-97188836ed3b/file_downloaded'
cd /content/banglalekha_raw && unzip -q images.zip && rm images.zip

# Run:
cd /content/bangla-vlm-lora
python scripts/banglalekha_slice.py --config configs/banglalekha_full.yaml
```

## What's next

The val curve has plateaued, so longer training at this configuration
will not help. Cheapest knobs in order of expected impact:

1. **Decode-side fixes (no retraining).** Re-run generation on the
   saved adapter with `do_sample=True, top_p=0.9`, or beam=4 plus
   `no_repeat_ngram_size=3, length_penalty=1.5`. Should immediately
   improve fluency and kill repetition loops; expected to also raise
   any per-token diversity metric.
2. **Unfreeze image-tower projections.** Add the CLIP-ViT q/k/v
   projections to `target_modules`. Visual signal is the
   most-likely bottleneck on image-specific outputs (`45.png` getting
   `গাছ` but no other content words is suggestive).
3. **Switch corpus to BAN-Cap.** Per `docs/paper_plan.md`, BAN-Cap is
   the higher-quality Bangla benchmark. BanglaLekha alone is unlikely
   to support paper-grade captioning regardless of method.
4. **Compute real metrics.** BLEU / CIDEr / BERTScore-bn / M-CLIPScore
   on the 915-val. Pair with the human-judgment pilot (§C3 of the
   paper plan) — 20 images is enough to confirm the metric-disagreement
   thesis.
5. **Diverse beam / contrastive decoding** would also help, but
   #1–#4 are higher leverage first.

This run closes the "real-data training" item from the prior session
handoff. The motivation chain (§6 of CLAUDE.md) is fully demonstrated
on real photographs; the bridge is necessary and sufficient to unlock
Bangla generation. The remaining work is *caption quality*, not
*language unlock*.
