"""Evaluate STL-1 semantic camera priors against VGGT camera hypotheses.

Each principal silhouette is omitted from the volume that predicts it.  The
test therefore measures whether the named front/top/left/right axes provide a
useful cross-view initialization without claiming calibrated cameras.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.reconstruct_dataset_phase3 import _calibrated_grids
from pipeline.train_dataset_phases import _binary_iou, _boundary_f1


PRINCIPAL_VIEWS = ("front", "top", "lsv", "rsv")
ALL_VIEWS = ("front", "top", "iso", "lsv", "rsv")
EXPECTED_FORWARD = {
    "front": np.asarray((0.0, -1.0, 0.0)),
    "top": np.asarray((0.0, 0.0, -1.0)),
    "iso": np.asarray((-1.0, -1.0, -1.0)) / np.sqrt(3.0),
    "lsv": np.asarray((1.0, 0.0, 0.0)),
    "rsv": np.asarray((-1.0, 0.0, 0.0)),
}


def _records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["object_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _read_mask(path: Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0


def _pairwise_angles(directions: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(ALL_VIEWS), len(ALL_VIEWS)), dtype=np.float64)
    for row, first in enumerate(ALL_VIEWS):
        for column, second in enumerate(ALL_VIEWS):
            cosine = np.clip(
                float(np.dot(directions[first], directions[second])), -1.0, 1.0
            )
            matrix[row, column] = np.degrees(np.arccos(cosine))
    return matrix


def _vggt_directions(extrinsic: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for index, view in enumerate(ALL_VIEWS):
        direction = extrinsic[index, :, :3].T @ np.asarray((0.0, 0.0, 1.0))
        result[view] = direction / max(float(np.linalg.norm(direction)), 1e-9)
    return result


def _holdout_volume(grids: dict[str, np.ndarray], omitted: str) -> np.ndarray:
    size = grids["front"].shape[0]
    volume = np.ones((size, size, size), dtype=bool)  # z, y, x
    if omitted != "front":
        volume &= grids["front"][:, None, :]
    if omitted != "top":
        volume &= grids["top"][None, :, :]
    sides = []
    if omitted != "lsv":
        sides.append(grids["lsv"])
    if omitted != "rsv":
        sides.append(np.fliplr(grids["rsv"]))
    if sides:
        side = np.logical_or.reduce(sides)
        volume &= side[:, :, None]
    return volume


def _project(volume: np.ndarray, view: str) -> np.ndarray:
    if view == "front":
        return volume.any(axis=1)
    if view == "top":
        return volume.any(axis=0)
    side = volume.any(axis=2)
    return np.fliplr(side) if view == "rsv" else side


def _overlay(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    image = np.full((*target.shape, 3), 245, dtype=np.uint8)
    both = target & prediction
    image[both] = (65, 170, 70)
    image[prediction & ~target] = (65, 65, 220)
    image[target & ~prediction] = (220, 130, 60)
    return image


def _cell(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(np.uint8(image) * 255, cv2.COLOR_GRAY2BGR)
    header = np.full((50, image.shape[1], 3), 248, dtype=np.uint8)
    cv2.putText(header, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (18, 18, 18), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(header, subtitle, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (70, 70, 70), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def validate_object(
    root: Path,
    record: dict[str, Any],
    vggt_dir: Path,
    output_dir: Path,
    resolution: int,
) -> dict[str, Any]:
    source_masks = {
        view: _read_mask(root / record["views"][view]["jewelry_mask_path"], 224)
        for view in ALL_VIEWS
    }
    grids, alignment = _calibrated_grids(source_masks, resolution)
    view_metrics = []
    rows = []
    for view in PRINCIPAL_VIEWS:
        volume = _holdout_volume(grids, view)
        prediction = _project(volume, view)
        target = grids[view]
        metrics = {
            "view": view,
            "iou": round(_binary_iou(target, prediction), 6),
            "boundary_f1_2px": round(_boundary_f1(target, prediction), 6),
            "volume_voxels_without_target": int(volume.sum()),
        }
        view_metrics.append(metrics)
        rows.append(
            np.hstack(
                (
                    _cell(target, f"{view.upper()} TARGET", "omitted from fit"),
                    _cell(prediction, "AXIS HOLDOUT", "other views only"),
                    _cell(
                        _overlay(target, prediction),
                        "AGREEMENT",
                        f"IoU {metrics['iou']:.3f}",
                    ),
                )
            )
        )

    cameras = np.load(vggt_dir / "vggt_cameras.npz")
    predicted_directions = _vggt_directions(cameras["extrinsic"][0])
    expected_angles = _pairwise_angles(EXPECTED_FORWARD)
    predicted_angles = _pairwise_angles(predicted_directions)
    angle_error = np.abs(predicted_angles - expected_angles)
    upper = np.triu_indices(len(ALL_VIEWS), k=1)
    mean_angle_error = float(np.mean(angle_error[upper]))
    max_angle_error = float(np.max(angle_error[upper]))

    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / f"{record['object_id']}_semantic_camera_holdout.png"
    if not cv2.imwrite(str(preview), np.vstack(rows)):
        raise RuntimeError(f"Could not write {preview}")
    mean_iou = float(np.mean([item["iou"] for item in view_metrics]))
    minimum_iou = float(min(item["iou"] for item in view_metrics))
    result = {
        "schema_version": "semantic_camera_prior_validation_v1",
        "object_id": record["object_id"],
        "split": record["split"],
        "resolution": resolution,
        "vggt_camera_diagnostic": {
            "view_order": list(ALL_VIEWS),
            "expected_pairwise_optical_axis_angles_deg": np.round(expected_angles, 3).tolist(),
            "predicted_pairwise_optical_axis_angles_deg": np.round(predicted_angles, 3).tolist(),
            "absolute_error_deg": np.round(angle_error, 3).tolist(),
            "mean_pairwise_error_deg": round(mean_angle_error, 6),
            "maximum_pairwise_error_deg": round(max_angle_error, 6),
        },
        "semantic_axis_holdout": {
            "views": view_metrics,
            "mean_iou": round(mean_iou, 6),
            "minimum_iou": round(minimum_iou, 6),
            "axis_alignment": alignment,
        },
        "checks": {
            "all_principal_views_omitted_and_scored": len(view_metrics) == 4,
            "no_omitted_mask_used_as_volume_constraint": True,
            "semantic_prior_beats_vggt_reprojection_floor": mean_iou >= 0.30,
            "semantic_prior_ready_as_initialization": mean_iou >= 0.70 and minimum_iou >= 0.50,
        },
        "decision": (
            "accepted_as_uncalibrated_camera_initialization"
            if mean_iou >= 0.70 and minimum_iou >= 0.50
            else "rejected_as_camera_initialization"
        ),
        "artifacts": {"holdout_preview": str(preview.relative_to(root))},
        "claims": {
            "camera_calibration_validated": False,
            "metric_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "View names are treated as orthographic axis hypotheses, not calibrated camera measurements.",
            "Shared-axis scale fitting uses silhouette bounding boxes, including the omitted target extent.",
            "Opposite side evidence remains available when one side is omitted.",
            "The isometric view is used only in the VGGT camera-angle diagnostic, not the axis holdout score.",
        ],
    }
    (output_dir / "semantic_camera_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run(root: Path, vggt_root: Path, output_root: Path, resolution: int) -> dict[str, Any]:
    run_report = json.loads((vggt_root / "run_report.json").read_text(encoding="utf-8"))
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    results = [
        validate_object(
            root,
            records[item["object_id"]],
            vggt_root / item["object_id"],
            output_root / item["object_id"],
            resolution,
        )
        for item in run_report["objects"]
    ]
    summary = {
        "schema_version": "semantic_camera_prior_multiring_v1",
        "dataset": "STL-1",
        "object_count": len(results),
        "objects": [
            {
                "object_id": item["object_id"],
                "split": item["split"],
                "vggt_mean_pairwise_angle_error_deg": item["vggt_camera_diagnostic"]["mean_pairwise_error_deg"],
                "semantic_holdout_mean_iou": item["semantic_axis_holdout"]["mean_iou"],
                "semantic_holdout_minimum_iou": item["semantic_axis_holdout"]["minimum_iou"],
                "decision": item["decision"],
                "artifacts": item["artifacts"],
            }
            for item in results
        ],
        "cohort": {
            "mean_vggt_pairwise_angle_error_deg": round(
                float(np.mean([item["vggt_camera_diagnostic"]["mean_pairwise_error_deg"] for item in results])), 6
            ),
            "mean_semantic_holdout_iou": round(
                float(np.mean([item["semantic_axis_holdout"]["mean_iou"] for item in results])), 6
            ),
            "minimum_semantic_holdout_iou": round(
                float(min(item["semantic_axis_holdout"]["minimum_iou"] for item in results)), 6
            ),
            "accepted_initialization_count": sum(
                item["decision"] == "accepted_as_uncalibrated_camera_initialization"
                for item in results
            ),
        },
        "decision": "research_only",
        "claims": {
            "camera_calibration_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--vggt-root",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_224_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/semantic_camera_priors_v1"),
    )
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    print(
        json.dumps(
            run(root, resolve(args.vggt_root), resolve(args.output_dir), args.resolution),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
