# Smoke test: GiT-base + LoRA on synthetic Bangla

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/smoke_git_lora.py`
**Config:** `configs/smoke_git_lora.yaml`
**Companion:** `docs/results/smoke_git_base/README.md` (full-FT baseline)

## Setup

- Model: `microsoft/git-base` wrapped with PEFT LoRA
- LoRA: rank 8, alpha 16, dropout 0.05, target `["query", "key", "value"]`
  (text-side BERT attention; CLIP-side and output layers untouched)
- Dataset: same 16 in-memory synthetic samples as the full-FT smoke
- 50 steps, batch 4, AdamW lr 1e-4 (higher than full-FT — only adapter trains)
- PEFT 0.19.1; **no `task_type` set** in `LoraConfig` (see notes below)

## Result

| Metric | LoRA value | Full-FT comparison |
|---|---|---|
| Trainable parameters | **221,184 (0.125%)** | 176,840,250 (100%) |
| Steps | 50 | 50 |
| Loss start | 12.35 | 12.29 |
| Loss end | 10.06 | 5.06 |
| Wall time | **7.7 s** | 16.6 s |
| Adapter / model size | **1.57 MB** | ~700 MB |
| Reference caption | `একটি লাল গাড়ি রাস্তায় চলছে।` | (same) |
| Generated output | **`part of a wall`** | `##াাাাাাাাাাাাাাাাাাাা` |
| Status | **PASS** | PASS |

## Findings worth carrying into the paper

### 1. The HF peft #1958 concern does not reproduce here

The HuggingFace discussion notes that PEFT LoRA on GiT "doesn't learn"
when `task_type=CAUSAL_LM` is set. With peft 0.19.1, **omitting
`task_type` and listing target modules explicitly** (`query, key, value`)
yields a clean wrap and a measurable loss decrease (12.35 → 10.06 over
50 steps on 16 in-memory samples). We document this as the recommended
recipe; setting `task_type` should be tested in an ablation and
documented if it reproduces the failure mode.

### 2. LoRA on attention alone is insufficient for cross-lingual adaptation

This is the most paper-worthy finding from the smoke test. Compared to
full FT, which produced `##াাাাা...` (the WordPiece continuation token
plus the Bangla vowel sign *aa* — the only Bangla codepoint in
`bert-base-uncased`), the LoRA model produces fluent **English**:
`part of a wall`. The mechanism is straightforward:

- LoRA on `query/key/value` adapts attention computation only.
- Input embeddings and the LM head remain frozen at their pretrained
  English-only values.
- Output token distribution is therefore unchanged — the model literally
  cannot raise the probability of Bangla tokens it has never seen.

This is the qualitative companion to the Section 3 tokenizer audit and
sharpens contribution C1: **a tokenizer/embedding bridge is mandatory
for LoRA, not merely an enhancement**. Attention-only LoRA on an
English VLM cannot produce Bangla, period.

### 3. Cost profile is what you expect

- Trainable params: 0.125% of the base.
- Step throughput is roughly 2× full FT (no gradient through frozen weights).
- Adapter on disk: 1.57 MB vs ~700 MB for the full model — checkpointing
  for many ablation runs is essentially free.

## Caveats

- Synthetic dataset; no quantitative claim about real Bangla captioning.
- Only one LoRA configuration tested (rank 8, q/k/v). Rank and
  target-module ablations belong in the real-data experiments.
- 50 steps is far too few for fair evaluation — point here is solely
  that the pipeline behaves and the qualitative output is informative.

## Reproduce

```bash
python scripts/smoke_git_lora.py --config configs/smoke_git_lora.yaml
```
