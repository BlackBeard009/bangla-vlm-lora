# Tokenizer fertility audit

Corpus: **2000** Bangla sentences.

| Tokenizer | Role | Vocab | BN vocab | Fertility ↓ | UNK% ↓ | Round-trip ↑ |
|---|---|---|---|---|---|---|
| microsoft/git-base | VLM (primary target) — English BERT WordPiece | 30522 | 72 | 4.880 | 2.50% | 77.37% |
| Salesforce/blip-image-captioning-base | VLM — English BERT WordPiece | 30522 | 72 | 4.880 | 2.50% | 77.37% |
| microsoft/Florence-2-base | VLM — BART (English) | 50265 | 0 | 14.167 | 0.00% | 100.00% |
| Salesforce/blip2-opt-2.7b | VLM — OPT (English) | 50266 | 0 | 14.167 | 0.00% | 100.00% |
| Qwen/Qwen2-VL-2B-Instruct | VLM — multilingual baseline | 151657 | 0 | 7.534 | 0.00% | 100.00% |
| facebook/mbart-large-50 | Multilingual decoder (Bangla supported) | 250054 | 2499 | 2.180 | 0.06% | 99.91% |
| csebuetnlp/banglabert | Bangla-native upper bound (encoder) | 32000 | 29127 | 1.345 | 0.73% | 97.99% |
| csebuetnlp/banglat5 | Bangla-native upper bound (seq2seq) | 32100 | 28644 | 1.352 | 0.81% | 99.78% |
