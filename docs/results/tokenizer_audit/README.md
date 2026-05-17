# Tokenizer fertility audit — Section 3 of the paper

**Updated:** 2026-05-17 (added bridged GiT row; results re-run on same corpus)
**Script:** `scripts/tokenizer_audit.py`
**Config:** `configs/tokenizer_audit.yaml`
**Corpus:** 2,000 Bangla Wikipedia sentences (`wikimedia/wikipedia 20231101.bn`, sentence-split on `।`).

Table file (regenerated each run): [`tokenizer_audit_results.md`](tokenizer_audit_results.md) · [`tokenizer_audit_results.csv`](tokenizer_audit_results.csv).

## What this audit measures

For each candidate tokenizer:

- **Vocab** — `len(tokenizer)`.
- **BN vocab** — count of vocab entries containing at least one codepoint in `U+0980..U+09FF`.
- **Fertility ↓** — mean subtokens per whitespace-word; lower is better. Captures how badly the tokenizer fragments Bangla.
- **UNK% ↓** — share of subtokens that resolve to the tokenizer's UNK id.
- **Round-trip ↑** — `1 − mean CER(decode(encode(x)), x)`. Lower means the normalizer or decoder is destroying information.

## Headline finding

| Tokenizer | BN vocab | Fertility | UNK% | Round-trip |
|---|---:|---:|---:|---:|
| microsoft/git-base (target VLM) | 72 | **4.880** | 2.50% | 77.37% |
| **+ BanglaBERT bridge (C1)** | **29,127** | **1.333** | **0.50%** | **97.17%** |
| csebuetnlp/banglabert (upper bound) | 29,127 | 1.345 | 0.73% | 97.99% |

The bridge closes — and in fertility/UNK%, slightly surpasses — the gap to
a Bangla-native tokenizer:

- **Fertility 4.88 → 1.33** (3.66× reduction). Bridged GiT actually beats
  BanglaBERT on this metric (1.333 vs 1.345) because it retains 30,522
  English tokens that are useful for the loanwords, English names, and
  digits that appear in Bangla Wikipedia.
- **UNK rate 2.50% → 0.50%** (5× reduction). The residual 0.50% is mostly
  rare characters that BanglaBERT also UNKs (0.73%).
- **Round-trip 77.4% → 97.2%** (+19.8 pp). Note this required the
  normalization fix in commit `dba58dc`; with the BERT defaults
  (`do_lower_case=True`, `strip_accents=None`), the bridged tokenizer
  stripped Bangla nukta/virama/candrabindu and the donor wordpieces
  could not be matched. See `docs/results/git_bridged_ablation/README.md`
  § "Honest caveats" #3 for the before/after token breakdown.

## What else the table shows

- **GiT-base ≡ BLIP-image-captioning-base** in every metric — both wrap
  `bert-base-uncased`. Don't double-count this finding in the paper.
- **Florence-2 (BART) and BLIP-2 (OPT) report `bn_vocab=0` and fertility
  ≈ 14.** The zero is a measurement artifact: those byte-level BPEs
  represent Bangla as byte fragments, none of which contain a Bangla
  codepoint in their *token string*. The fertility ≈ 14 is the real
  story — each Bangla character costs 3 UTF-8 bytes, each of which
  becomes a separate BPE token. Round-trip 100% confirms the bytes are
  preserved; the cost is purely sequence length.
- **Qwen2-VL-2B (multilingual baseline) gets fertility 7.53.** Better
  than monolingual English VLMs (14×) but far worse than mbart's 2.18 or
  the bridged GiT's 1.33. Its 151k-token vocab still allocates almost
  nothing to Bangla.
- **mbart-large-50 at fertility 2.18** is the strongest off-the-shelf
  multilingual decoder for Bangla. The bridge brings GiT below this
  number while keeping GiT's smaller vocab (~60k vs mbart's 250k).
- **BanglaT5 fertility 1.35** is essentially tied with BanglaBERT, as
  expected — both are Bangla-native WordPiece/SentencePiece models.
- **PaliGemma still missing.** Gated model; needs one-time
  `huggingface-cli login` with accepted license.

## Reproduce

```bash
python scripts/tokenizer_audit.py --config configs/tokenizer_audit.yaml
```

Wall time: ~1–2 min on Colab (CPU is fine — tokenizers only). Writes
both files to `paths.EXP_RESULTS/tokenizer_audit/`; copy the headline
artifacts here when results change.

## What this implies for the paper

Section 3 of `docs/paper_plan.md` can now make a quantitative claim
that contribution C1 (the tokenizer/embedding bridge) is **not just
qualitatively necessary** (Bangla emerges in `git_bridged_ablation`)
but also **measurably brings the VLM's Bangla tokenizer to parity with
a Bangla-native tokenizer** — at 1.333 fertility, vs. 1.345 for
BanglaBERT, on the same corpus.
