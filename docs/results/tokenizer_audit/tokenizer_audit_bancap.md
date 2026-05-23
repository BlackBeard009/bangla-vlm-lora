# Tokenizer fertility audit

Corpus: **2000** Bangla sentences.

| Tokenizer | Role | Vocab | BN vocab | Fertility ↓ | UNK% ↓ | Round-trip ↑ |
|---|---|---|---|---|---|---|
| microsoft/git-base | VLM (primary target) — English BERT WordPiece | 30522 | 72 | 4.155 | 2.05% | 78.20% |
| microsoft/git-base + csebuetnlp/banglabert (bridged) | VLM — bridged: GiT vocab + BanglaBERT wordpieces (C1) | 59577 | 29127 | 1.176 | 5.92% | 91.18% |
| Salesforce/blip-image-captioning-base | VLM — English BERT WordPiece | 30522 | 72 | 4.155 | 2.05% | 78.20% |
| microsoft/Florence-2-base | VLM — BART (English) | 50265 | 0 | 11.622 | 0.00% | 100.00% |
| Salesforce/blip2-opt-2.7b | VLM — OPT (English) | 50266 | 0 | 11.622 | 0.00% | 100.00% |
| Qwen/Qwen2-VL-2B-Instruct | VLM — multilingual baseline | 151657 | 0 | 6.287 | 0.00% | 97.08% |
| facebook/mbart-large-50 | Multilingual decoder (Bangla supported) | 250054 | 2499 | 1.976 | 0.00% | 97.05% |
| csebuetnlp/banglabert | Bangla-native upper bound (encoder) | 32000 | 29127 | 1.176 | 5.92% | 91.18% |
| csebuetnlp/banglat5 | Bangla-native upper bound (seq2seq) | 32100 | 28644 | 1.335 | 5.94% | 98.52% |
