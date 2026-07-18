"""Run the SemMargin metric through the trivial-baseline gauntlet.

Scores every system with saved BAN-Cap predictions under the
margin-normalized semantic similarity metric
(src/evaluation/semantic_margin.py) plus a held-out-reference human
ceiling. Pass criteria for the metric (not the systems):

  1. trivial no-vision baselines score ~0 margin,
  2. real systems rank GiT < Qwen-bancap <= curriculum,
  3. the 0%-Bangla zero-shot row does NOT score above real Bangla
     systems (the M-CLIPScore language-blindness test),
  4. the human ceiling clears every system.

Usage:
    python scripts/semantic_margin_gauntlet.py --config configs/semantic_margin_gauntlet.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from src.evaluation.semantic_margin import (  # noqa: E402
    human_ceiling,
    semantic_margin,
)


def load_predictions(subdir: str, variant: str) -> tuple[list, list, list]:
    p = paths.EXP_RESULTS / subdir / "predictions.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    preds = d["predictions"]
    if isinstance(preds, dict):
        preds = preds[variant]
    return d["filenames"], d["references"], preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_name = cfg["model_name"]

    rows = []
    base_filenames = None
    base_refs = None
    t0 = time.time()
    for sys_cfg in cfg["systems"]:
        name = sys_cfg["name"]
        filenames, refs, hyps = load_predictions(
            sys_cfg["results_subdir"], sys_cfg.get("variant", "zero_shot")
        )
        if base_filenames is None:
            base_filenames, base_refs = filenames, refs
        elif filenames != base_filenames:
            raise RuntimeError(f"{name}: val split misaligned with first system")
        scores = semantic_margin(hyps, refs, model_name=model_name)
        rows.append({"system": name, **scores})
        print(f"[{time.time() - t0:6.0f}s] {name}: "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    ceiling = human_ceiling(base_refs, model_name=model_name)
    rows.append({"system": "HUMAN (held-out ref)", **ceiling})
    print(f"[{time.time() - t0:6.0f}s] HUMAN (held-out ref): "
          + "  ".join(f"{k}={v:.4f}" for k, v in ceiling.items()))

    out = paths.EXP_RESULTS / cfg["output"]["results_subdir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"model_name": model_name, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    lines = [
        "# SemMargin gauntlet — margin-normalized semantic similarity",
        "",
        f"Encoder: `{model_name}` · BAN-Cap val (809 images x 5 refs)",
        "",
        "| System | raw | prior | **margin** | rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['sem_raw']:.4f} | {r['sem_prior']:.4f} "
            f"| **{r['sem_margin']:.4f}** | {r['sem_rank']:.4f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
