"""Single source of truth for filesystem paths.

Resolution order for the artifact root:
1. `BANGLA_VLM_ROOT` env var (local workstation, e.g. D:\\bangla-vlm-lora)
2. Google Drive mount (Colab)
3. `_local_data/` inside the repo (gitignored scratch fallback)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DRIVE_ROOT = Path("/content/drive/MyDrive/bangla-vlm-lora")
LOCAL_FALLBACK = REPO_ROOT / "_local_data"

_env_root = os.environ.get("BANGLA_VLM_ROOT")
if _env_root:
    ARTIFACT_ROOT = Path(_env_root)
elif DRIVE_ROOT.exists():
    ARTIFACT_ROOT = DRIVE_ROOT
else:
    ARTIFACT_ROOT = LOCAL_FALLBACK

DATA_RAW = ARTIFACT_ROOT / "data" / "raw"
DATA_PROCESSED = ARTIFACT_ROOT / "data" / "processed"
DATA_HUMAN_EVAL = ARTIFACT_ROOT / "data" / "human_eval"

EXP_CHECKPOINTS = ARTIFACT_ROOT / "experiments" / "checkpoints"
EXP_LOGS = ARTIFACT_ROOT / "experiments" / "logs"
EXP_RESULTS = ARTIFACT_ROOT / "experiments" / "results"


def ensure_dirs() -> None:
    for p in [
        DATA_RAW, DATA_PROCESSED, DATA_HUMAN_EVAL,
        EXP_CHECKPOINTS, EXP_LOGS, EXP_RESULTS,
    ]:
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Artifact root: {ARTIFACT_ROOT}")
    for p in [DATA_RAW, DATA_PROCESSED, DATA_HUMAN_EVAL,
              EXP_CHECKPOINTS, EXP_LOGS, EXP_RESULTS]:
        print(f"  {p}  (exists={p.exists()})")
