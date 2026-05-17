"""Single source of truth for filesystem paths.

Edit `DRIVE_ROOT` once Google Drive is mounted to switch from local testing
to persistent storage. Everything else derives from it.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Flip to "/content/drive/MyDrive/bangla-vlm-lora" once Drive is mounted.
DRIVE_ROOT = Path("/content/drive/MyDrive/bangla-vlm-lora")
LOCAL_FALLBACK = REPO_ROOT / "_local_data"

ARTIFACT_ROOT = DRIVE_ROOT if DRIVE_ROOT.exists() else LOCAL_FALLBACK

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
