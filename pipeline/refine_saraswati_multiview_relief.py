"""Fuse confidence-gated front and top depth into a blind visual hull.

Side-view depth is deliberately excluded when the mirrored left/right audit
fails.  Each proposal is frozen before post-hoc access to the supplied OBJ.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pipeline.reconstruct_dataset_phase3 import _fit_volume, _mesh_from_volume
    from pipeline.refine_saraswati_blind_relief import _normalized_depth, _warp_centered_values
    from pipeline.run_saraswati_blind_benchmark import (
        DEFAULT_GROUND_TRUTH,
        DEFAULT_IMAGES,
        _foreground_mask,
        _sha256,
        _write_json,
        evaluate_against_withheld_obj,
    )
    from pipeline.sweep_saraswati_relief import _metric_vector, _non_regressing
except ModuleNotFoundError:  # Direct ``python pipeline/...`` execution.
    from reconstruct_dataset_phase3 import _fit_volume, _mesh_from_volume
    from refine_saraswati_blind_relief import _normalized_depth, _warp_centered_values
    from run_saraswati_blind_benchmark import (
        DEFAULT_GROUND_TRUTH,
        DEFAULT_IMAGES,
        _foreground_mask,
        _sha256,
        _write_json,
        evaluate_against_withheld_obj,
    )
    from sweep_saraswati_relief import _metric_vector, _non_regressing


DEFAULT_BASELINE = Path("dataset/phase_runs/v1/saraswati_blind_benchmark_256_v1")
DEFAULT_OUTPUT = Path("dataset/phase_runs/v1/saraswati_blind_multiview_relief_v1")


def carve_axis_relief(
    volume: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    view: str,
    strength: float,
    direction: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inset the camera-facing surface while preserving two occupied voxels."""
    if view not in {"front", "top", "lsv", "rsv"}:
        raise ValueError(f"unsupported view: {view}")
    if direction not in {"low", "high"}:
        raise ValueError("direction must be low or high")
    if not 0 < strength <= 0.2:
        raise ValueError("strength must be in (0, 0.2]")
    carved = volume.copy()
    removed = affected = 0
    for row, column in zip(*np.where(valid)):
        if view == "front":
            occupied = np.flatnonzero(carved[row, :, column])
        elif view == "top":
            occupied = np.flatnonzero(carved[:, row, column])
        else:
            occupied = np.flatnonzero(carved[row, column, :])
        if occupied.size < 3:
            continue
        thickness = int(occupied[-1] - occupied[0] + 1)
        inset = min(
            int(round(float(depth[row, column]) * strength * thickness)),
            max(0, thickness - 2),
        )
        if not inset:
            continue
        start, end = (
            (int(occupied[0]), int(occupied[0] + inset))
            if direction == "low"
            else (int(occupied[-1] - inset + 1), int(occupied[-1] + 1))
        )
        if view == "front":
            carved[row, start:end, column] = False
        elif view == "top":
            carved[start:end, row, column] = False
        else:
            carved[row, column, start:end] = False
        removed += end - start
        affected += 1
    return carved, {
        "view": view,
        "direction": direction,
        "strength": strength,
        "removed_voxels": removed,
        "affected_columns": affected,
    }


def _depth_hypotheses(
    baseline_root: Path,
    masks: dict[str, np.ndarray],
    fit: dict[str, Any],
    resolution: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    audit = json.loads((baseline_root / "depth_audit.json").read_text(encoding="utf-8"))
    backend = json.loads(
        (baseline_root / "depth_backends/depth_anything_v2_metric/backend_report.json").read_text(encoding="utf-8")
    )
    selected = []
    excluded = []
    hypotheses = {}
    for view in ("front", "top", "lsv", "rsv"):
        eligible = audit["views"][view]["eligible_as_relative_relief_hypothesis"]
        if view in {"lsv", "rsv"} and audit["cross_view"]["left_right_mirrored_depth_correlation"] < 0.25:
            eligible = False
        if not eligible:
            excluded.append(view)
            continue
        record = next(item for item in backend["records"] if item["view"] == view)
        normalized, _ = _normalized_depth(np.load(record["artifacts"]["depth_m"]), masks[view])
        scale = float(fit["axis_alignment"]["warp_scales_input_px_to_voxel"][view])
        warped, valid = _warp_centered_values(normalized, masks[view], scale, resolution)
        if view == "rsv":
            warped, valid = np.fliplr(warped), np.fliplr(valid)
        hypotheses[view] = (warped, valid)
        selected.append(view)
    return hypotheses, {
        "selected_views": selected,
        "excluded_views": excluded,
        "left_right_correlation": audit["cross_view"]["left_right_mirrored_depth_correlation"],
    }


def run_sweep(
    baseline_root: Path,
    output_root: Path,
    ground_truth: Path,
    strengths: list[float],
    resolution: int = 256,
) -> Path:
    masks = {view: _foreground_mask(path)[0] for view, path in DEFAULT_IMAGES.items()}
    base_volume, side_mode, fit, _ = _fit_volume(masks, resolution)
    hypotheses, evidence = _depth_hypotheses(baseline_root, masks, fit, resolution)
    baseline = _metric_vector(json.loads((baseline_root / "evaluation_report.json").read_text(encoding="utf-8")))
    candidates = []
    for direction in ("low", "high"):
        for strength in strengths:
            volume = base_volume.copy()
            operations = []
            for view in evidence["selected_views"]:
                volume, operation = carve_axis_relief(
                    volume, *hypotheses[view], view, strength, direction
                )
                operations.append(operation)
            candidate_root = output_root / f"{direction}_{strength:.3f}"
            candidate_root.mkdir(parents=True, exist_ok=True)
            mesh = _mesh_from_volume(volume, nominal_size_mm=50.0)
            proposal = candidate_root / "multiview_relief_proposal.obj"
            mesh.export(proposal)
            record = {
                "schema_version": "image2cad_multiview_relief_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ground_truth_accessed": False,
                "ground_truth_path_recorded": False,
                "method": "confidence_gated_multi_axis_relative_depth_carve",
                "selected_side_consensus": side_mode,
                "evidence": evidence,
                "operations": operations,
                "proposal": {
                    "path": str(proposal),
                    "sha256": _sha256(proposal),
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "watertight": bool(mesh.is_watertight),
                },
            }
            record_path = candidate_root / "reconstruction_record.json"
            _write_json(record_path, record)
            evaluation = evaluate_against_withheld_obj(proposal, record_path, ground_truth, candidate_root)
            metrics = _metric_vector(json.loads(evaluation.read_text(encoding="utf-8")))
            candidates.append({
                "direction": direction,
                "strength": strength,
                "metrics": metrics,
                "non_regressing": _non_regressing(metrics, baseline),
                "proposal": str(proposal),
            })
    accepted = [item for item in candidates if item["non_regressing"]]
    report = {
        "schema_version": "image2cad_multiview_relief_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "evidence_gate": evidence,
        "candidates": candidates,
        "decision": "accept_best_non_regressing" if accepted else "retain_visual_hull_baseline",
        "selected": min(accepted, key=lambda item: item["metrics"]["chamfer_l1_normalized"], default=None),
        "manufacturing_accuracy_validated": False,
    }
    path = output_root / "sweep_report.json"
    _write_json(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.01, 0.02, 0.04, 0.06])
    args = parser.parse_args()
    print(run_sweep(args.baseline, args.output, args.ground_truth, args.strengths))


if __name__ == "__main__":
    main()
