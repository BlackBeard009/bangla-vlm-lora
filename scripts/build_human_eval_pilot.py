"""Build the §C3 human-evaluation pilot annotation pack.

Samples N images from the BAN-Cap val split (seeded — NOT selected by
model performance; cherry-picking images would invalidate the eval),
pulls each system's caption from its best decode variant, blinds and
shuffles the (image, caption) rows, and writes:

  paths.DATA_HUMAN_EVAL/pilot/
    annotation.csv   — what annotators fill in (no system identities)
    key.csv          — row-id -> system mapping (do NOT give to annotators)
    images/          — copies of the sampled images
    README.md        — annotator instructions (adequacy + fluency, 1-5)

Usage:
    python scripts/build_human_eval_pilot.py --config configs/human_eval_pilot.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_predictions(subdir: str, variant: str) -> dict:
    d = json.loads(
        (paths.EXP_RESULTS / subdir / "predictions.json").read_text(
            encoding="utf-8")
    )
    preds = d["predictions"]
    if isinstance(preds, dict):
        preds = preds[variant]
    return {
        "filenames": d["filenames"],
        "references": d["references"],
        "predictions": preds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rng = random.Random(cfg["seed"])

    systems = {}
    base = None
    for s in cfg["systems"]:
        data = load_predictions(s["results_subdir"], s.get("variant", "zero_shot"))
        if base is None:
            base = data
        elif data["filenames"] != base["filenames"]:
            raise RuntimeError(f"{s['name']}: split misaligned")
        systems[s["name"]] = data["predictions"]

    n_images = cfg["n_images"]
    picked = rng.sample(range(len(base["filenames"])), n_images)

    out = paths.DATA_HUMAN_EVAL / cfg["output_subdir"]
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    images_dir = resolve_path(cfg["images_dir"])

    rows = []
    for idx in picked:
        fname = base["filenames"][idx]
        shutil.copy2(images_dir / fname, img_out / fname)
        pairs = [(name, systems[name][idx]) for name in systems]
        rng.shuffle(pairs)  # independent caption order per image
        for name, caption in pairs:
            rows.append({"filename": fname, "system": name,
                         "caption": caption.strip()})

    with (out / "annotation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "image", "caption",
                    "adequacy_1to5", "fluency_1to5", "comments"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["filename"], r["caption"], "", "", ""])

    with (out / "key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "image", "system"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["filename"], r["system"]])

    (out / "README.md").write_text(
        "\n".join([
            "# Bangla caption human evaluation — pilot",
            "",
            f"{n_images} images x {len(systems)} captions = {len(rows)} rows.",
            "Rate every row in `annotation.csv`. The image files are in",
            "`images/`. Do not skip rows; do not discuss ratings with",
            "other annotators; the caption sources are hidden on purpose.",
            "",
            "## Adequacy (1-5) — does the caption describe THIS image?",
            "- 5: fully correct, covers the main content",
            "- 4: correct with minor omissions",
            "- 3: partially correct (some right, some wrong/missing)",
            "- 2: mostly wrong but topically related",
            "- 1: unrelated to the image",
            "",
            "## Fluency (1-5) — is it good Bangla, ignoring the image?",
            "- 5: fluent, natural Bangla",
            "- 4: minor awkwardness",
            "- 3: understandable but clearly flawed",
            "- 2: broken grammar, hard to understand",
            "- 1: not understandable / not Bangla",
            "",
            "Judge each caption on its own; scores are absolute, not",
            "relative to the other captions of the same image.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} annotation rows to {out}")
    print("Give annotators: annotation.csv + images/ + README.md")
    print("Keep key.csv private until scoring.")


if __name__ == "__main__":
    main()
