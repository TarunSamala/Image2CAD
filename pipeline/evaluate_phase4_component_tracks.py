"""Evaluate Ring01 prong identities against retained joint VGGT point maps.

Front-view prong IDs are the reference.  The stage computes robust non-metric
3D centroids inside each resolved mask, exhaustively evaluates all 24 target
permutations, and reports whether the independently best VGGT assignment agrees
with the image-plane proposal.  It does not convert machine agreement into
human truth, physical scale, or CAD authorization.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")
RESOLVED_VIEWS = ("front", "top", "angled")
COMPONENT_IDS = tuple(f"prong_{index:03d}" for index in range(1, 5))
COLORS_BGR = (
    (56, 56, 230),
    (42, 174, 255),
    (210, 120, 40),
    (180, 70, 190),
)


def _square_mask(path: Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    height, width = mask.shape
    side = max(height, width)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = mask
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST) > 0


def _square_image(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    side = max(height, width)
    canvas = np.full((side, side, 3), 250, dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = image
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)


def _descriptor(
    mask: np.ndarray,
    world_points: np.ndarray,
    confidence: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    finite = mask & np.isfinite(world_points).all(axis=-1) & np.isfinite(confidence)
    values = confidence[finite]
    if not len(values):
        raise ValueError("Component mask has no finite VGGT evidence")
    threshold = float(np.quantile(values, 0.25))
    selected = finite & (confidence >= threshold)
    points = world_points[selected]
    if len(points) < 8:
        raise ValueError(f"Component has only {len(points)} retained VGGT points")
    centre = np.median(points, axis=0)
    distances = np.linalg.norm(points - centre, axis=1)
    return (
        {
            "point_count": int(len(points)),
            "confidence_q25_threshold": round(threshold, 6),
            "confidence_median": round(float(np.median(confidence[selected])), 6),
            "robust_centroid_xyz": [round(float(value), 8) for value in centre],
            "median_cluster_radius": round(float(np.median(distances)), 8),
            "p90_cluster_radius": round(float(np.quantile(distances, 0.90)), 8),
            "units": "vggt_model_relative_non_metric",
        },
        centre,
    )


def _object_diagonal(
    root: Path,
    manifest: dict[str, Any],
    world: np.ndarray,
    confidence: np.ndarray,
    size: int,
) -> float:
    samples = []
    for view in RESOLVED_VIEWS:
        index = VIEWS.index(view)
        path = root / manifest["views"][view]["proposals"]["jewelry"]["path"]
        mask = _square_mask(path, size)
        finite = mask & np.isfinite(world[index]).all(axis=-1) & np.isfinite(confidence[index])
        threshold = np.quantile(confidence[index][finite], 0.25)
        samples.append(world[index][finite & (confidence[index] >= threshold)])
    points = np.concatenate(samples, axis=0)
    lower, upper = np.percentile(points, (1, 99), axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Invalid VGGT object diagonal")
    return diagonal


def _assignment(distance: np.ndarray) -> dict[str, Any]:
    candidates = []
    for permutation in itertools.permutations(range(len(COMPONENT_IDS))):
        cost = float(sum(distance[index, permutation[index]] for index in range(len(permutation))))
        candidates.append((cost, permutation))
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_cost, best = candidates[0]
    second_cost, second = candidates[1]
    identity = tuple(range(len(COMPONENT_IDS)))
    identity_cost = float(sum(distance[index, index] for index in range(len(COMPONENT_IDS))))
    return {
        "best_assignment": {
            COMPONENT_IDS[index]: COMPONENT_IDS[target]
            for index, target in enumerate(best)
        },
        "best_assignment_indices": [int(index + 1) for index in best],
        "best_cost_fraction": round(best_cost, 8),
        "second_best_assignment_indices": [int(index + 1) for index in second],
        "second_best_cost_fraction": round(second_cost, 8),
        "permutation_margin_fraction": round(
            (second_cost - best_cost) / max(second_cost, 1e-12), 8
        ),
        "identity_assignment_cost_fraction": round(identity_cost, 8),
        "identity_is_geometry_optimal": best == identity,
    }


def _evidence_level(view: str, record: dict[str, Any]) -> str:
    median = record["median_identity_distance_fraction"]
    margin = record["permutation_margin_fraction"]
    if record["identity_is_geometry_optimal"] and median <= 0.05 and margin >= 0.25:
        return "strong_machine_support"
    if record["identity_is_geometry_optimal"] and median <= 0.15 and margin >= 0.10:
        return "moderate_machine_support"
    return "insufficient_or_contradictory"


def _label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = np.full((48, image.shape[1], 3), 250, dtype=np.uint8)
    cv2.putText(header, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(header, subtitle, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (75, 75, 75), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def _mask_panel(
    root: Path,
    manifest: dict[str, Any],
    instances: dict[str, Any],
    view: str,
    size: int,
) -> np.ndarray:
    source = _square_image(root / manifest["views"][view]["source"]["path"], size)
    overlay = source.copy()
    for index, component_id in enumerate(COMPONENT_IDS):
        path = root / instances["views"][view]["component_observations"][component_id]["visible_mask"]
        mask = _square_mask(path, size)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.30
            + np.asarray(COLORS_BGR[index], dtype=np.float32) * 0.70
        ).astype(np.uint8)
        centre = np.mean(np.argwhere(mask), axis=0)
        cv2.putText(
            overlay,
            str(index + 1),
            (int(centre[1]) - 4, int(centre[0]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _matrix_panel(view: str, matrix: np.ndarray, result: dict[str, Any], size: int = 320) -> np.ndarray:
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    origin = 72
    cell = 52
    cv2.putText(canvas, f"FRONT -> {view.upper()}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "normalized centroid distance", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (70, 70, 70), 1, cv2.LINE_AA)
    for index in range(4):
        cv2.putText(canvas, f"T{index + 1}", (origin + index * cell + 14, origin - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"F{index + 1}", (25, origin + index * cell + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 40), 1, cv2.LINE_AA)
        for target in range(4):
            value = float(matrix[index, target])
            intensity = int(np.clip(value / 0.30, 0, 1) * 210)
            color = (255 - intensity // 3, 245 - intensity // 2, 235 - intensity)
            p0 = (origin + target * cell, origin + index * cell)
            p1 = (p0[0] + cell - 2, p0[1] + cell - 2)
            cv2.rectangle(canvas, p0, p1, color, -1)
            thickness = 2 if index == target else 1
            border = COLORS_BGR[index] if index == target else (205, 205, 205)
            cv2.rectangle(canvas, p0, p1, border, thickness)
            cv2.putText(canvas, f"{value:.3f}", (p0[0] + 5, p0[1] + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, result["evidence_level"].replace("_", " "), (8, size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 90, 30), 1, cv2.LINE_AA)
    return canvas


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    instance_path = root / "data/phase4_prong_instances_v1/report.json"
    dense_path = root / "dataset/phase_runs/v1/ring01_vggt_dense_224_v1/vggt_dense_evidence.npz"
    vggt_report_path = root / "dataset/phase_runs/v1/ring01_vggt_dense_224_v1/vggt_report.json"
    manifest = json.loads((root / "data/ring01_ground_truth_v1/manifest.json").read_text(encoding="utf-8"))
    instances = json.loads(instance_path.read_text(encoding="utf-8"))
    vggt_report = json.loads(vggt_report_path.read_text(encoding="utf-8"))
    with np.load(dense_path) as dense:
        world = dense["world_points"][0]
        confidence = dense["world_points_conf"][0]
    size = int(vggt_report["image_size"])
    diagonal = _object_diagonal(root, manifest, world, confidence, size)
    descriptors: dict[str, Any] = {}
    centres: dict[str, dict[str, np.ndarray]] = {}
    for view in RESOLVED_VIEWS:
        view_index = VIEWS.index(view)
        descriptors[view] = {}
        centres[view] = {}
        for component_id in COMPONENT_IDS:
            observation = instances["views"][view]["component_observations"][component_id]
            mask = _square_mask(root / observation["visible_mask"], size)
            descriptor, centre = _descriptor(mask, world[view_index], confidence[view_index])
            descriptors[view][component_id] = descriptor
            centres[view][component_id] = centre

    reference = np.stack([centres["front"][component_id] for component_id in COMPONENT_IDS])
    comparisons: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    for view in ("top", "angled"):
        target = np.stack([centres[view][component_id] for component_id in COMPONENT_IDS])
        matrix = np.linalg.norm(reference[:, None, :] - target[None, :, :], axis=2) / diagonal
        matrices[view] = matrix
        assignment = _assignment(matrix)
        diagonal_values = np.diag(matrix)
        record = {
            **assignment,
            "distance_matrix_fraction": [
                [round(float(value), 8) for value in row] for row in matrix
            ],
            "identity_distance_fraction_by_component": {
                component_id: round(float(diagonal_values[index]), 8)
                for index, component_id in enumerate(COMPONENT_IDS)
            },
            "mean_identity_distance_fraction": round(float(np.mean(diagonal_values)), 8),
            "median_identity_distance_fraction": round(float(np.median(diagonal_values)), 8),
            "maximum_identity_distance_fraction": round(float(np.max(diagonal_values)), 8),
        }
        record["evidence_level"] = _evidence_level(view, record)
        comparisons[view] = record

    checks = {
        "retained_vggt_run_is_numerically_complete": vggt_report["status"] == "completed_machine_hypothesis",
        "all_resolved_masks_have_vggt_points": all(
            descriptors[view][component_id]["point_count"] >= 8
            for view in RESOLVED_VIEWS
            for component_id in COMPONENT_IDS
        ),
        "top_identity_is_geometry_optimal": comparisons["top"]["identity_is_geometry_optimal"],
        "angled_identity_is_geometry_optimal": comparisons["angled"]["identity_is_geometry_optimal"],
        "top_has_strong_machine_support": comparisons["top"]["evidence_level"] == "strong_machine_support",
        "angled_has_at_least_moderate_machine_support": comparisons["angled"]["evidence_level"] in {"strong_machine_support", "moderate_machine_support"},
        "side_back_ambiguity_preserved": instances["summary"]["ambiguous_views"] == ["side", "back"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_cells = [
        _label(
            _mask_panel(root, manifest, instances, view, size),
            f"{view.upper()} PRONG IDS",
            "stable colours; machine proposals",
        )
        for view in RESOLVED_VIEWS
    ]
    top_matrix = _matrix_panel("top", matrices["top"], comparisons["top"])
    angled_matrix = _matrix_panel("angled", matrices["angled"], comparisons["angled"])
    top_row = cv2.hconcat(mask_cells)
    matrix_row = cv2.hconcat((top_matrix, angled_matrix))
    if matrix_row.shape[1] < top_row.shape[1]:
        matrix_row = cv2.copyMakeBorder(matrix_row, 0, 0, 0, top_row.shape[1] - matrix_row.shape[1], cv2.BORDER_CONSTANT, value=(250, 250, 250))
    elif top_row.shape[1] < matrix_row.shape[1]:
        top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, matrix_row.shape[1] - top_row.shape[1], cv2.BORDER_CONSTANT, value=(250, 250, 250))
    audit_path = output_dir / "ring01_component_track_audit.png"
    cv2.imwrite(str(audit_path), cv2.vconcat((top_row, matrix_row)))

    report = {
        "schema_version": "lalitha_phase4_component_track_evidence_v1",
        "object_id": "ring01",
        "execution_profile": "images_only",
        "authority": "machine_evidence_review_required",
        "reference_view": "front",
        "resolved_views": list(RESOLVED_VIEWS),
        "ambiguous_views": ["side", "back"],
        "component_ids": list(COMPONENT_IDS),
        "backend": {
            "name": "VGGT-1B joint world-point maps",
            "image_size": size,
            "metric": False,
            "checkpoint_sha256": vggt_report["checkpoint_sha256"],
            "pinned_code_commit": vggt_report["pinned_code_commit"],
            "license_id": vggt_report["checkpoint_license_id"],
            "source_dense_evidence": str(dense_path.relative_to(root)),
        },
        "normalization": {
            "object_diagonal": round(diagonal, 8),
            "units": "vggt_model_relative_non_metric",
            "bounds": "p01_to_p99_over_front_top_angled_jewellery_points",
        },
        "descriptors": descriptors,
        "comparisons": comparisons,
        "checks": checks,
        "gate": {
            "resolved_identity_assignments_machine_supported": all(checks.values()),
            "side_back_individual_identity_resolved": False,
            "cross_view_identity_reviewed": False,
            "metric_geometry_validated": False,
            "geometry_generation_authorized": False,
        },
        "artifacts": {
            "audit": str(audit_path.relative_to(root)),
            "instance_report": str(instance_path.relative_to(root)),
        },
        "claims": {
            "cross_view_identity_human_validated": False,
            "vggt_camera_accuracy_validated": False,
            "physical_correspondence_validated": False,
            "hidden_prong_geometry_recovered": False,
            "metric_scale_validated": False,
        },
        "limitations": [
            "VGGT agreement is independent machine evidence, not human ground truth.",
            "The 224 px point maps are non-metric and use uncalibrated product images.",
            "Angled-view support is moderate and weaker than top-view support.",
            "Side and back projected overlaps remain unresolved by design.",
            "This checkpoint does not create or validate CAD geometry.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase4_component_tracks_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(root, output_dir)
    print(json.dumps({"comparisons": report["comparisons"], "checks": report["checks"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
