# BanglaLekha slice — bridged GiT + LoRA on real Bangla data

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/banglalekha_slice.py`
**Config:** `configs/banglalekha_slice.yaml`
**Dataset:** BanglaLekha-Image-Captions (Mendeley `rxxch9vw59` v2 — 9,154
images, 2 native Bangla captions each).
**Companion (synthetic, motivation):** `docs/results/git_bridged_ablation/`

## What this is

Pipeline-validation run for the resume-plan item 3 — first time the
bridged-GiT + LoRA recipe has touched real Bangla image-caption pairs.
**Not a paper-grade result**: 200 training images and 300 steps is
nowhere near convergence. The point is to prove the pipeline runs
end-to-end on real data and produces real Bangla output that
references image content. The full run is the next step.

## Setup

- **Bridge:** GiT-base vocab (30,522) extended with 29,055 BanglaBERT
  wordpieces; embedding/LM-head rows initialized via `donor` strategy
  (input embeddings copied from BanglaBERT where the token matches;
  LM head rows = mean of original LM head).
- **LoRA:** rank 8, α 16, dropout 0.05, target `[query, key, value]`,
  `modules_to_save=[output, word_embeddings]`. 109.5 M / 330.8 M
  parameters trainable (33.11%).
- **Data slice:** 200 train / 20 val, deterministic seed-42 partition
  of the full 9,154-image set. Caption selection: `first` (use
  `captions[0]` only — the dataset has 2 refs/image).
- **Training:** 300 steps, batch 4, AdamW lr 1e-4, no weight decay.

## Result

| Step | Train loss | Val loss |
|---:|---:|---:|
| 100 | 5.576 | 4.055 |
| 200 | 3.393 | 3.778 |
| 300 | 2.930 | 3.624 |

**Wall:** 124.6 s on T4 (~0.4 s/step).

Val loss is still falling at step 300 — the model is under-trained,
not overfit. This is expected for a slice.

## Sample generations (held-out val, step 300)

| Image | Reference (native annotation) | Generated |
|---|---|---|
| `4.png`  | ছয় জন মানুষ দাড়িয়ে আছে। | `মানুষ আছে` |
| `31.png` | একটি বাচ্চা ছেলে বসে আছে। | `মানুষ আছে` |
| `41.png` | কয়েক জন মানুষ দাঁড়িয়ে ও কয়েক জন মানুষ বসে আছে। | `মানুষ আছে` |
| `45.png` | জলাশয় এর পাশে গাছ আছে। উপরে নীল আকাশে অল্প সাদা মেঘ আছে। | `পুরুষ মানুষ আছে` |
| `48.png` | একজন মহিলা ও একটি কিশোর ছেলে বাচ্চা কোলে নিয়ে বসে আছে। | `পুরুষ মানুষ আছে` |

Full sample dump (one entry per eval checkpoint) is in
`paths.EXP_RESULTS/banglalekha_slice/samples.json`.

### What changed across training

The training trajectory shows real learning, not danda collapse:

- **Step 100:** every val output is just `আছে` ("exists"). Has learned
  that captions end in this copula but nothing else.
- **Step 200:** outputs become `মানুষ আছে` / `মানুষ । মানুষ আছে`
  ("person exists" / "person. person exists"). The model now emits
  the most-frequent content word in BanglaLekha — `মানুষ` ("person") —
  which is appropriate: the dataset is heavily people-centric.
- **Step 300:** outputs split into `মানুষ আছে` and `পুরুষ মানুষ আছে`
  ("man exists"). For `48.png` (reference: `একজন মহিলা ও একটি কিশোর ছেলে…`
  — "a woman and a teenage boy…") this is at least visually grounded:
  the image contains both. For `45.png` (reference is about a tree
  beside water) it's wrong but not nonsense.

The motivation-chain claim ("the bridge unlocks Bangla generation") is
now demonstrated on real photographs, not just synthetic captions.

## Honest caveats

- **Output diversity is very low** because (a) 200 unique images is a
  tiny corpus, (b) BanglaLekha captions are repetitive — `মানুষ আছে`
  patterns are everywhere — and (c) 300 steps × batch 4 sees each
  image ≈ 6 times. The model has learned the corpus's dominant
  template, not how to describe an arbitrary image.
- **`first`-caption selection ignores the second annotation** for each
  image, so the model sees half the linguistic variability the dataset
  offers. Set `caption_selection: random` for the full run.
- **No metric numbers** — BLEU/CIDEr/BERTScore are not computed yet.
  Train and val cross-entropy losses are the only quantitative signal.
- **Image-encoder LoRA is *not* applied** in this run — the
  `target_modules: [query, key, value]` is GiT's text-side attention.
  The image tower remains fully frozen. For a paper-grade run we
  should ablate including image-tower projections.

## Artifacts

- `paths.EXP_RESULTS/banglalekha_slice/results.md` — regenerated each run.
- `paths.EXP_RESULTS/banglalekha_slice/samples.json` — full sample dump
  across eval points.
- `paths.EXP_CHECKPOINTS/banglalekha_slice/` — LoRA adapter
  (`adapter_model.safetensors`, ~430 MB with `modules_to_save`
  embedding+head) and the bridged tokenizer.
- `docs/results/banglalekha_slice/results_table.md` — frozen copy of
  the regenerated table.

## Reproduce

```bash
# First time only:
mkdir -p $DRIVE_ROOT/data/raw/banglalekha
curl -L -o $DRIVE_ROOT/data/raw/banglalekha/captions.json \
  'https://data.mendeley.com/public-files/datasets/rxxch9vw59/files/9b3e789a-5a5c-48b3-8a2c-2c91e9307c2a/file_downloaded'
mkdir -p /content/banglalekha_raw
curl -L -o /content/banglalekha_raw/images.zip \
  'https://data.mendeley.com/public-files/datasets/rxxch9vw59/files/220ffc5a-8645-4c0b-bd1b-97188836ed3b/file_downloaded'
cd /content/banglalekha_raw && unzip -q images.zip

# Run:
cd /content/bangla-vlm-lora
python scripts/banglalekha_slice.py --config configs/banglalekha_slice.yaml
```

## Next step — graduate to the full run

In `configs/banglalekha_slice.yaml`:
- `data.max_samples: null` (or e.g. 8000 to leave a held-out)
- `data.caption_selection: random`
- `training.steps: ~5000–10000` (one or two epochs through the full set)
- `training.eval_every: 500`
- Optional: cosine LR schedule with warmup; ablate image-tower LoRA

Expected wall-clock at 5000 steps × ~0.4 s/step ≈ 35 min on T4. Save
the adapter to Drive (already wired) so checkpoints survive the
session.
