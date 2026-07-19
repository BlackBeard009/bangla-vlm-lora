"""Export the human-eval pilot pack into the annotation SPA's public/ dir.

Reads paths.DATA_HUMAN_EVAL/pilot/annotation.csv (blinded — no system
identities) and the copied images, writes annotation_app/public/
{data.json, images/}. Rerun after rebuilding the pilot pack.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

pilot = paths.DATA_HUMAN_EVAL / "pilot"
public = REPO_ROOT / "annotation_app" / "public"
(public / "images").mkdir(parents=True, exist_ok=True)

rows = []
with (pilot / "annotation.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        rows.append({
            "row_id": int(r["row_id"]),
            "image": r["image"],
            "caption": r["caption"],
        })

(public / "data.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

n = 0
for img in (pilot / "images").iterdir():
    shutil.copy2(img, public / "images" / img.name)
    n += 1

print(f"Wrote {len(rows)} rows to {public / 'data.json'}; copied {n} images")
