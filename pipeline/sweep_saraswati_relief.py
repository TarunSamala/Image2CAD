"""Benchmark bounded image-depth relief strengths against the accepted hull.

Every candidate is reconstructed before the withheld OBJ is opened.  The OBJ
is used only by the post-hoc evaluator, and a candidate is accepted only when
it does not regress any retained surface metric.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from pipeline.refine_saraswati_blind_relief import refine
    from pipeline.run_saraswati_blind_benchmark import (
        DEFAULT_GROUND_TRUTH,
        _write_json,
        evaluate_against_withheld_obj,
    )
except ModuleNotFoundError:  # Direct ``python pipeline/...`` execution.
    from refine_saraswati_blind_relief import refine
    from run_saraswati_blind_benchmark import (
        DEFAULT_GROUND_TRUTH,
        _write_json,
        evaluate_against_withheld_obj,
    )


DEFAULT_BASELINE = Path("dataset/phase_runs/v1/saraswati_blind_benchmark_256_v1")
DEFAULT_OUTPUT = Path("dataset/phase_runs/v1/saraswati_blind_relief_sweep_v1")


def _metric_vector(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["surface_metrics"]
    scores = metrics["surface_f_scores"]
    return {
        "chamfer_l1_normalized": float(metrics["chamfer_l1_normalized"]),
        "hausdorff_95_normalized": float(metrics["hausdorff_95_normalized"]),
        "f_score_1pct": float(scores["1pct"]["f_score"]),
        "f_score_2pct": float(scores["2pct"]["f_score"]),
        "f_score_5pct": float(scores["5pct"]["f_score"]),
    }


def _non_regressing(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        candidate["chamfer_l1_normalized"] <= baseline["chamfer_l1_normalized"]
        and candidate["hausdorff_95_normalized"] <= baseline["hausdorff_95_normalized"]
        and all(candidate[key] >= baseline[key] for key in (
            "f_score_1pct", "f_score_2pct", "f_score_5pct"
        ))
    )


def sweep(
    baseline_root: Path,
    output_root: Path,
    ground_truth: Path,
    strengths: list[float],
    resolution: int = 256,
) -> Path:
    baseline_report = json.loads(
        (baseline_root / "evaluation_report.json").read_text(encoding="utf-8")
    )
    baseline = _metric_vector(baseline_report)
    candidates = []
    for strength in strengths:
        candidate_root = output_root / f"strength_{strength:.3f}"
        proposal, record = refine(
            baseline_root, candidate_root, strength=strength, resolution=resolution
        )
        evaluation_path = evaluate_against_withheld_obj(
            proposal, record, ground_truth, candidate_root
        )
        metrics = _metric_vector(
            json.loads(evaluation_path.read_text(encoding="utf-8"))
        )
        candidates.append(
            {
                "strength": strength,
                "metrics": metrics,
                "non_regressing": _non_regressing(metrics, baseline),
                "proposal": str(proposal),
                "evaluation": str(evaluation_path),
            }
        )

    accepted = [item for item in candidates if item["non_regressing"]]
    report = {
        "schema_version": "image2cad_blind_relief_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "blind_reconstruction_contract": True,
        "baseline": baseline,
        "candidates": candidates,
        "decision": "accept_best_non_regressing" if accepted else "retain_visual_hull_baseline",
        "selected": min(
            accepted,
            key=lambda item: item["metrics"]["chamfer_l1_normalized"],
            default=None,
        ),
        "limits": [
            "This sweep tunes one relative-depth carving method on one held-out ring.",
            "Selection does not establish generalization or manufacturing accuracy.",
        ],
    }
    report_path = output_root / "sweep_report.json"
    _write_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.06, 0.10]
    )
    args = parser.parse_args()
    print(sweep(args.baseline, args.output, args.ground_truth, args.strengths, args.resolution))


if __name__ == "__main__":
    main()
