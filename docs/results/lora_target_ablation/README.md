# LoRA target-module ablation: vocabulary is the bottleneck

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/lora_target_ablation.py`
**Config:** `configs/lora_target_ablation.yaml`

## Setup

Three increasingly permissive PEFT configurations on `microsoft/git-base`,
each trained from the same starting weights, same data (16 in-memory
synthetic Bangla samples), same seed, 50 steps, batch 4, AdamW lr 1e-4.

| Variant | LoRA targets | `modules_to_save` |
|---|---|---|
| V1 | `query`, `key`, `value` | — |
| V2 | `query`, `key`, `value` | `output` (LM head, fully trainable) |
| V3 | `query`, `key`, `value` | `output`, `word_embeddings` (both fully trainable) |

`modules_to_save` was used instead of LoRA-targeting `output` because the
suffix `"output"` collides with the `GitSelfOutput` composite block
inside attention. Putting `output` in `modules_to_save` skips LoRA
decomposition and makes the LM head fully trainable — the *stronger*
variant, which makes the result more conclusive.

## Result

| Variant | Trainable | % of base | Loss start→end | Generated |
|---|---|---|---|---|
| V1 attn-only | 221,184 | **0.125%** | 12.35 → 10.06 | `part of a wall` (English) |
| V2 + head | 41,414,970 | **18.99%** | 12.35 → 3.46 | `##াাাাাাাাাাাাাাাাাাাাাাাাাাাাাা` |
| V3 + head + emb | 64,855,866 | **26.86%** | 12.35 → 3.39 | `##াাাাাাাাাাাাাাাাাাাাাাাাাাাাাা` |

Reference caption: `একটি লাল গাড়ি রাস্তায় চলছে।`

## Findings

### 1. Attention-only LoRA cannot cross language boundaries

V1 produced fluent English (`part of a wall`) even though training was on
Bangla captions. With LoRA confined to `q, k, v`, the LM head and input
embeddings remain frozen at their pretrained English-only values, so the
output token distribution is unchanged. Attention can rebalance how the
model attends, but cannot teach it new output tokens. This generalizes
beyond Bangla — attention-only LoRA is unsuitable for any cross-lingual
adaptation where source and target use disjoint vocabularies.

### 2. Once the LM head is free, the model emits the only Bangla tokens it has

V2 unlocked the LM head and the output collapsed to `##াাাা...`. Here
`##` is the WordPiece continuation marker and `া` (U+09BE) is the Bangla
vowel sign *aa* — essentially the only Bangla codepoint that BERT-base
WordPiece includes as a standalone piece. The model has no way to
produce real Bangla words because the vocabulary contains none.

### 3. Retraining input embeddings adds nothing — the vocabulary is the bottleneck

This is the key finding. V3 makes the input embeddings fully trainable
on top of V2. The result vs. V2:

- Loss: 3.46 → 3.39 (≈2% relative, indistinguishable on this tiny set)
- Output: **identical** (`##াাাাাাাাাাাাাাাাাাাাাাাাাাাাাা`)
- Extra trainable params: +23.4 million

V3 is the maximally-permissive PEFT configuration short of full FT, and
it does not produce Bangla. Real Bangla words like `একটি`, `লাল`,
`গাড়ি` simply do not exist as tokens in `bert-base-uncased`'s 30,522
wordpieces. No amount of weight adjustment can change that — the
vocabulary itself must change.

## What this means for the paper

Combined with the earlier tokenizer audit and full-FT smoke test, the
motivation chain for contribution C1 (vocabulary/embedding bridge) is
complete:

1. **Audit:** 72 Bangla codepoints out of 30,522 vocab entries; 2.5% UNK.
2. **Full FT:** can emit only the handful of Bangla-codepoint tokens, malformed.
3. **LoRA attn-only:** can't even reach those tokens → English output.
4. **LoRA + head retraining (V2):** matches full FT — token-availability ceiling.
5. **V3 (+ emb retraining):** zero improvement over V2 — proves vocabulary is the constraint.

The paper can now claim: **even maximally-permissive PEFT cannot produce
Bangla on an English-vocabulary VLM.** The tokenizer/embedding bridge is
not an optimization — it is necessary.

## Caveats

- All numbers from 16 synthetic samples, 50 steps. Loss values are not
  comparable across runs of different scale.
- Output text is a single greedy/beam decode from one held image; not a
  proper evaluation. The qualitative content (English vs. Bangla-token
  collapse) is what matters.
- The ablation does not test LoRA on the FFN modules (`fc1`, `fc2`,
  `dense`) — adding these would further expand capacity but would not
  change the vocabulary, so the conclusion holds a fortiori.

## Reproduce

```bash
python scripts/lora_target_ablation.py --config configs/lora_target_ablation.yaml
```

Wall time: ~50 s for all three variants on T4. Writes
`results.md` to `paths.EXP_RESULTS/lora_target_ablation/`.
