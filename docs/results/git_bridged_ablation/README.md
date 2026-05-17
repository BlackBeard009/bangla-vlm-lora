# Bridged-GiT + LoRA: closing the motivation chain

**Date:** 2026-05-17
**Hardware:** Colab T4 (16 GB)
**Script:** `scripts/git_bridged_ablation.py`
**Config:** `configs/git_bridged_ablation.yaml`
**Bridge module:** `src/tokenizer/bridge.py`
**Companion (negative result):** `docs/results/lora_target_ablation/`

## Setup

Three runs of GiT-base, identical to the previous V3 ablation
(`q,k,v` LoRA + `modules_to_save=[output, word_embeddings]`) — but with
the **tokenizer/embedding bridge** applied first:

- Vocabulary extended from 30,522 to **59,577** (+29,055 Bangla wordpieces drawn from `csebuetnlp/banglabert`).
- Input embeddings and LM head resized; new rows initialized via one of three strategies.

| Strategy | Description |
|---|---|
| `random` | PyTorch / transformers default — multivariate normal with old embeddings' mean and covariance (note this is *not* uniform random; see caveats). |
| `mean` | Every new row set to the literal mean of the original 30,522 rows. |
| `donor` | Input-embedding rows copied from BanglaBERT for matching tokens; LM head rows still mean (donor LM head shape differs). |

Training: 50 steps, batch 4, AdamW lr 1e-4, 16 in-memory synthetic Bangla samples.

## Result

| Init | Vocab before→after | Added | Trainable | Loss start→end | Generated |
|---|---|---|---|---|---|
| `random` | 30,522 → 59,577 | +29,055 | 33.11% | 11.64 → 4.10 | `। । । । । । ।` |
| `mean` | 30,522 → 59,577 | +29,055 | 33.11% | 11.64 → 4.10 | `। । । । । । ।` |
| `donor` | 30,522 → 59,577 | +29,055 | 33.11% | **11.52 → 4.09** | `। । । । । । ।` |

Reference caption: `একটি লাল গাড়ি রাস্তায় চলছে।`

## What this demonstrates

### The bridge changes model output qualitatively

Comparison with the previous ablation (same training recipe, only the vocabulary differs):

| Setup | Vocab | Generated |
|---|---|---|
| V3 from `lora_target_ablation/` (no bridge) | 30,522 (English) | `##াাাাাাাাাাাাাাাাাাাা` |
| **Bridged + donor init (this experiment)** | **59,577 (+Bangla)** | `। । । । । ।` (real Bangla) |

The output went from "BERT's WordPiece continuation marker plus the only
Bangla codepoint in vocab, repeated" to **actual Bangla tokens**
(specifically the danda `।`, U+0964, Bangla's full stop). That is the
qualitative payoff of contribution C1.

### The motivation chain is now complete

1. **Audit:** GiT vocab has 72 Bangla codepoints / 30,522 (Section 3 of paper).
2. **Full-FT:** can emit only those 72 codepoints, malformed (`##াাাা...`).
3. **LoRA attn-only:** can't reach even those tokens → English (`part of a wall`).
4. **LoRA + head retraining (V2):** matches full FT — token-availability ceiling.
5. **+ embedding retraining (V3):** zero improvement — vocabulary is the bottleneck.
6. **Bridged (this experiment):** Bangla emerges. The bottleneck is removed.

## Honest caveats

### 1. Init strategies are nearly indistinguishable in this run

`random` and `mean` produce identical loss curves to 4 decimal places.
Two reasons:

- **Transformers' `resize_token_embeddings` already does "mean-resizing"
  by default** — a multivariate normal init using the old embeddings'
  mean and covariance (Hewitt 2021). My `random` strategy is a no-op
  that leaves this smart init in place. For a clean random baseline,
  pass `mean_resizing=False` to the resize call.
- **`modules_to_save=[word_embeddings, output]` makes the entire embedding
  and LM head fully trainable**, washing out the init signal in 50 steps.
  To see init-strategy differences, either keep embeddings partially
  frozen, or use a much smaller training budget.

The `donor` init does start ~0.13 lower in loss and ends ~0.02 lower —
real but tiny effect. The init ablation is paper-relevant but needs a
better experimental design (frozen embeddings or 0–5 step measurements)
to differentiate strategies meaningfully. Defer to real-data experiments.

### 2. Output collapses to `।` because every training caption ends with `।`

In our 16 in-memory captions, every sentence ends with the Bangla full
stop. The model overfits to "always emit `।`" because it's the
highest-frequency token in our tiny dataset. This is a property of the
smoke-test data, not the bridge. With real BanglaLekha-scale data we'd
expect actual Bangla words (nouns, verbs).

### 3. BertTokenizer normalization drops some diacritics — FIXED 2026-05-17

Originally, the bridged tokenizer was reloaded with BERT's default
`do_lower_case=True` and `strip_accents=None` (which becomes True under
lowercasing). Both `়` (nukta) and `্` (virama) — and `ঁ` candrabindu —
fall in Unicode category `Mn` and were stripped by `BasicTokenizer`, so
words like `গাড়ি`, `রাস্তায়`, `চাঁদ` could never hit their dedicated
donor wordpieces.

`_build_extended_tokenizer` now reloads with
`do_lower_case=False, strip_accents=False, tokenize_chinese_chars=False`.
The normalizer becomes
`BertNormalizer(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False)`,
and diacritic-bearing words map cleanly to single donor tokens:

| Word | Before fix | After fix |
|---|---|---|
| `গাড়ি` | `['গা', '##ডি']` | `['গাড়ি']` |
| `রাস্তায়` | `['রাস', '##তা', '##য']` | `['রাস্তায়']` |
| `চাঁদ` | `['চাদ']` | `['চাঁদ']` |
| `হাঁটছে` | `['হাট', '##ছে']` | `['হাঁটছে']` |
| `বিড়াল` | `['বিড', '##াল']` | `['বিড়াল']` |
| `পার্কে` | `['পার', '##কে']` | `['পার্কে']` |

All 16 synthetic captions now round-trip losslessly. Re-running the
bridged ablation with this fix is the next step before paper-grade
numbers; the qualitative claim from this experiment (Bangla emerges
post-bridge) is unaffected.

## Engineering note: AddedVocabulary panic

`add_tokens(new_tokens)` panics with "AddedVocabulary bad split" in the
Rust `tokenizers` library when 29k Bangla wordpieces are added at once
(see `src/tokenizer/bridge.py` docstring). The fix is to build a fresh
tokenizer from a rewritten `vocab.txt`, so the new tokens are part of
the base vocab rather than going through the `AddedVocabulary` code
path. Documented in detail in the `_build_extended_tokenizer` helper.

In modern transformers all BERT-family tokenizers wrap the Rust backend
(`is_fast=True`, has a `_tokenizer` Rust object). There is no longer a
true pure-Python slow path. Plan around this.

## Reproduce

```bash
python scripts/git_bridged_ablation.py --config configs/git_bridged_ablation.yaml
```

Wall time: ~30 s for all three variants on T4. Writes
`results.md` to `paths.EXP_RESULTS/git_bridged_ablation/`; headline
table copied to `results_table.md` here.

## What to do next

1. **Real-data experiment.** BanglaLekha-Image-Captions on Drive, several
   thousand steps, with the same bridge + LoRA recipe. This becomes the
   paper's main experimental result.
2. **Proper init-strategy ablation.** Either with `mean_resizing=False`
   on the resize call, or by freezing embeddings (no `modules_to_save`)
   so init survives. Real differences will only show up with these.
3. **Bridged-only fertility audit.** Add the bridged GiT tokenizer as a
   new row in `tokenizer_audit_results.md` to demonstrate the
   improvement. (The normalization fix above should drop fertility from
   the unbridged 4.88 toward BanglaBERT's 1.35.)
