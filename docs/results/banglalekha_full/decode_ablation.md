# Decode-side ablation on the saved banglalekha_full adapter

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/decode_ablation.py`
**Config:** `configs/decode_ablation.yaml`
**Adapter:** `paths.EXP_CHECKPOINTS/banglalekha_full/` (bridged GiT +
LoRA, donor init, 5,000 steps — see `README.md` in this directory).

## What this is

A pure decode comparison. The headline `banglalekha_full` table was
produced with beam=4 and no length/repetition penalties — the same
decode used during training-time eval. This run reuses the saved
adapter (no retraining) and re-generates captions for the same eight
held-out val images under three decoding configurations.

Goal: separate "the model can't" from "the decoder won't." Item 1 of
the resume-point list in `SESSION_HANDOFF.md`.

## Variants

| Variant | Params |
|---|---|
| `baseline_beam4` | beam=4, max_new_tokens=30 (matches training-time eval) |
| `beam4_penalties` | beam=4 + `no_repeat_ngram_size=3`, `length_penalty=1.5`, `early_stopping=True` |
| `topp_sample` | `do_sample=True`, `top_p=0.9`, beam=1 |

All three load the same checkpoint. Seed re-applied before each
variant so `topp_sample` is reproducible across reruns.

## Sample comparisons (first 8 val items, deterministic seed=42)

| Image | Reference | `baseline_beam4` | `beam4_penalties` | `topp_sample` |
|---|---|---|---|---|
| `4.png` | ছয় জন মানুষ দাড়িয়ে আছে। | পুরুষ করছে | পুরুষ করছে পিছনে মানুষ করছে | শিশু । মানুষ করছে |
| `31.png` | একটি বাচ্চা ছেলে বসে আছে। | পুরুষ আছে | পুরুষ । একজন আছে | শরীরে পুরুষ করছে |
| `41.png` | কয়েক জন মানুষ দাঁড়িয়ে ও কয়েক জন মানুষ বসে আছে। | মানুষ করছে | মানুষ । একজন আছে | জন যাচ্ছেন মাঝে |
| `45.png` | জলাশয় এর পাশে গাছ আছে। উপরে নীল আকাশে অল্প সাদা মেঘ আছে। | । গাছ । একটি গাছ । । একটি গাছ । গাছ । একটি গাছ । আছে | । গাছ । একটি গাছ | আছে |
| `48.png` | একজন মহিলা ও একটি কিশোর ছেলে বাচ্চা কোলে নিয়ে বসে আছে। | শিশু আছে | শিশু । একজন । একটি আছে | ছেলে করছেন |
| `65.png` | সবুজ ফসলের মাঠে একজন বালক লাল জামা পরিধান করে আছে। | মানুষ করছে | মানুষ করছে | মানুষ । একজন । আছে |
| `71.png` | সবুজ ঘাসের উপর বালক, বালিকা ও শিশু আছে। | শিশু । একজন আছে | শিশু । একজন আছে | মহিলা থেকে । ও থেকে শিশু করছে |
| `110.png` | পানির ভিতর একজন বালক খালি শরীরে বসে আছে। | মানুষ । অনেকগুলো আছে | মানুষ । অনেকগুলো আছে | ছেলে একটি । একজন একটি দিচ্ছে আছে অনেকগুলো ছেলে |

## What we learn

1. **The danda-repetition loop on `45.png` is purely a decode artifact.**
   `no_repeat_ngram_size=3` cuts `। গাছ । একটি গাছ । । একটি গাছ । গাছ ।
   একটি গাছ । আছে` down to `। গাছ । একটি গাছ` — the trigram constraint
   is enough. This was the most visible "model is broken" output in
   the headline samples; it isn't a model bug.
2. **Length penalty alone barely changes lengths.** Most beam outputs
   stay at 2–4 wordpieces even with `length_penalty=1.5`. The model
   happily emits EOS early — a property of the corpus prior
   (BanglaLekha captions are short), not a decoder timidity.
3. **Sampling reveals lexical diversity that beam search hides.**
   `topp_sample` surfaces tokens that never appear in either beam
   variant: `শরীরে`, `যাচ্ছেন`, `মাঝে`, `ছেলে`, `করছেন`, `মহিলা`,
   `দিচ্ছে`, `পিছনে`. The model has learned more vocabulary than the
   beam outputs suggest — beam search collapses onto the high-prob
   prefix.
4. **Sampling sometimes hits more specific tokens.** `48.png` (ref:
   "a woman and a teenage boy") gets `ছেলে` ("boy") under sampling vs.
   `শিশু` ("child") under beam. `71.png` (ref includes
   `বালক, বালিকা ও শিশু`) gets `মহিলা ... শিশু` under sampling. These
   are still wrong on details, but they're using the right *category*
   of words for the image.
5. **Sampling is high-variance.** `45.png` collapses to just `আছে` under
   sampling — single-sample top-p is a coin flip per token. Multi-sample
   sampling with a reranker (CLIP-similarity, length, or a small Bangla
   LM) would smooth this out. Not done here; flagged as future work.

## Implications for the paper / next experiments

- **The "outputs are templated" claim from `README.md` is partly a
  decode artifact, partly a corpus prior.** The bridged model's
  output distribution is richer than the beam samples imply; beam=4
  with no penalties is essentially the worst-case decoder for
  diversity.
- **`no_repeat_ngram_size=3` is a free upgrade.** It eliminates the
  most embarrassing output (`45.png`) without retraining or changing
  anything else.
- **Real metrics need both decode settings.** When item #2 of the
  handoff (BLEU/CIDEr/BERTScore-bn/M-CLIPScore on the 915-val) lands,
  it should report at minimum `baseline_beam4` and a `beam_penalties`
  variant — otherwise BLEU-4 will look artificially bad because of
  the `45.png` failure mode and similar.
- **Image-tower LoRA (handoff item #3) and the BAN-Cap switch (item
  #4) remain the high-leverage retraining moves** — sampling
  diversifies vocabulary but doesn't add image-specific information
  that wasn't already in the encoder.

## Artifacts

- `paths.EXP_RESULTS/decode_ablation/results.md` — same table as above.
- `paths.EXP_RESULTS/decode_ablation/samples.json` — per-variant decode
  output, structured as `{variant_name: [{filename, reference, generated}]}`.
- `docs/results/banglalekha_full/decode_ablation.md` — this write-up.

## Reproduce

```bash
cd /content/bangla-vlm-lora
python scripts/decode_ablation.py --config configs/decode_ablation.yaml
```

Requires the `banglalekha_full` adapter at
`paths.EXP_CHECKPOINTS/banglalekha_full/` and the BanglaLekha
captions + images per the parent `README.md` Reproduce block.
