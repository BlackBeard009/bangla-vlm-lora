# bangla-vlm-lora

LoRA fine-tuning of Vision-Language Models for Bangla image captioning.

See `docs/paper_plan.md` for the research plan.

## Repository layout

```
bangla-vlm-lora/
├── configs/          # YAML run configs (one per experiment)
├── src/              # reusable Python modules (import as `from src... import ...`)
│   ├── data/         # dataset loaders, preprocessing
│   ├── models/       # base-model loaders, LoRA wiring
│   ├── tokenizer/    # tokenizer extension, embedding init strategies
│   ├── training/     # train loops, callbacks
│   ├── evaluation/   # metrics: n-gram, BERTScore, CLIPScore, Polos, etc.
│   └── utils/        # shared helpers (logging, seeding, paths)
├── scripts/          # thin CLI entry points calling into src/
├── notebooks/        # Colab-facing orchestration; logic lives in src/
├── docs/             # paper plan, design notes
└── paths.py          # single source of truth for data/checkpoint locations
```

## Starting a new Colab session

Open `notebooks/00_bootstrap.ipynb` from this repo (or just paste the four
cells) — it mounts Drive, clones/pulls this repo, installs dependencies, and
verifies that paths resolve. Run it first every session.

## Data and checkpoints live on Google Drive

Code is in this directory; large artifacts (datasets, model weights, results)
go to a mounted Drive folder. `paths.py` resolves both. To switch from local
testing to Drive, edit `paths.py` only.

Mount Drive at the start of each Colab session:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then expected Drive layout:

```
/content/drive/MyDrive/bangla-vlm-lora/
├── data/
│   ├── raw/              # BAN-Cap, BanglaLekha, BNATURE, XM3600-bn
│   ├── processed/        # tokenized / image-feature caches
│   └── human_eval/       # annotation CSVs for the human-judgment benchmark
└── experiments/
    ├── checkpoints/      # LoRA adapters per run (named by config)
    ├── logs/             # training logs, W&B exports
    └── results/          # generated captions + metric tables
```

## Running things

Scripts are the source of truth; notebooks just call them:

```bash
python scripts/tokenizer_audit.py --config configs/tokenizer_audit.yaml
python scripts/train.py            --config configs/git_base_lora.yaml
python scripts/evaluate.py         --config configs/git_base_lora.yaml
```
