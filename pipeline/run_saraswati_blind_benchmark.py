"""Blind four-view Saraswati reconstruction and post-hoc OBJ evaluation.

The reconstruction function accepts image paths only. The ground-truth OBJ is
opened exclusively by the evaluation function after the proposal mesh and its
immutable reconstruction record have been written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
import trimesh

try:
    from pipeline.reconstruct_dataset_phase3 import _fit_volume, _mesh_from_volume
except ModuleNotFoundError:  # Direct ``python pipeline/...`` execution.
    from reconstruct_dataset_phase3 import _fit_volume, _mesh_from_volume


DEFAULT_IMAGES = {
    "front": Path("dataset/STL-1/Ring 2/R2 - Front - Saraswati.png"),
    "top": Path("dataset/STL-1/Ring 2/R2 - Top.png"),
    "lsv": Path("dataset/STL-1/Ring 2/R2 - LSV.png"),
    "rsv": Path("dataset/STL-1/Ring 2/R2 - RSV.png"),
}
DEFAULT_GROUND_TRUTH = Path("dataset/Model Test 3d/Saraswati.obj")
DEFAULT_OUTPUT = Path("dataset/phase_runs/v1/saraswati_blind_benchmark_v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _foreground_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background[None, None, :], axis=2)
    scaled = np.clip(distance, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(12.0, float(otsu))
    mask = np.where(distance >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    minimum = max(16, int(mask.size * 0.00008))
    keep = np.zeros_like(mask)
    retained = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            keep[labels == label] = 255
            retained += 1
    if not np.any(keep):
        raise ValueError(f"foreground extraction produced an empty mask: {path}")
    return keep > 0, {
        "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
        "background_bgr": [round(float(value), 3) for value in background],
        "distance_threshold": round(threshold, 3),
        "mask_occupancy": round(float(np.mean(keep > 0)), 7),
        "retained_components": retained,
    }


def reconstruct_from_images(
    images: dict[str, Path],
    output_dir: Path,
    resolution: int = 128,
    *,
    object_id: str = "saraswati_blind",
    dataset_name: str = "STL-1/Ring 2",
    proposal_filename: str = "saraswati_image_only_proposal.obj",
) -> tuple[Path, Path]:
    """Create a proposal without accepting or opening any ground-truth mesh."""
    required = {"front", "top", "lsv", "rsv"}
    if set(images) != required:
        raise ValueError(f"exactly these four image semantics are required: {sorted(required)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    masks: dict[str, np.ndarray] = {}
    extraction: dict[str, Any] = {}
    source_records: dict[str, Any] = {}
    metric_depth_samples: list[dict[str, Any]] = []
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    for view in sorted(images):
        path = images[view].resolve()
        mask, metadata = _foreground_mask(path)
        masks[view] = mask
        mask_path = mask_dir / f"{view}.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        extraction[view] = {**metadata, "mask_path": str(mask_path)}
        source_records[view] = {"path": str(path), "sha256": _sha256(path)}
        metric_depth_samples.append(
            {
                "sample_key": f"{object_id}:{view}",
                "dataset": dataset_name,
                "object_id": object_id,
                "split": "blind_benchmark",
                "view": view,
                "image_path": str(images[view]),
                "mask_path": str(mask_path),
                "source_sha256": _sha256(path),
                "physical_scale": None,
                "camera_calibrated": False,
            }
        )

    metric_manifest_path = output_dir / "metric_depth_input_manifest.json"
    _write_json(
        metric_manifest_path,
        {
            "schema_version": "image2cad_metric_depth_consensus_input_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_profile": "image_only_four_view",
            "measurement_contract": "model_hypothesis_only_no_physical_scale",
            "object_count": 1,
            "sample_count": 4,
            "selected_objects": [object_id],
            "selection_reason": "four-view image-only reconstruction",
            "samples": metric_depth_samples,
        },
    )

    volume, side_mode, fit, normalized_masks = _fit_volume(masks, resolution)
    mesh = _mesh_from_volume(volume, nominal_size_mm=50.0)
    if Path(proposal_filename).name != proposal_filename or not proposal_filename.endswith(".obj"):
        raise ValueError("proposal filename must be a plain OBJ filename")
    mesh_path = output_dir / proposal_filename
    mesh.export(mesh_path)
    for view, mask in normalized_masks.items():
        cv2.imwrite(str(mask_dir / f"{view}_normalized.png"), mask.astype(np.uint8) * 255)

    record = {
        "schema_version": "image2cad_image_only_reconstruction_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "reconstruction",
        "input_contract": "four_images_only",
        "source_images": source_records,
        "ground_truth_accessed": False,
        "ground_truth_path_recorded": False,
        "reconstruction_method": "four_view_silhouette_visual_hull",
        "resolution": resolution,
        "side_candidate": side_mode,
        "mask_extraction": extraction,
        "metric_depth_input_manifest": str(metric_manifest_path),
        "fit": fit,
        "proposal": {
            "path": str(mesh_path),
            "sha256": _sha256(mesh_path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "extents_non_metric": [round(float(value), 6) for value in mesh.extents],
        },
        "claims": {
            "independent_image_to_3d_proposal": True,
            "metric_scale_recovered": False,
            "fine_relief_recovered": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    record_path = output_dir / "reconstruction_record.json"
    _write_json(record_path, record)
    return mesh_path, record_path


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    geometries = [value for value in loaded.geometry.values() if len(value.vertices) and len(value.faces)]
    if not geometries:
        raise ValueError(f"no triangle geometry found in {path}")
    mesh = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    mesh.remove_unreferenced_vertices()
    return mesh


def _normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    centered = points - center
    scale = float(np.ptp(centered, axis=0).max())
    if scale <= 0:
        raise ValueError("cannot normalize zero-extent points")
    return centered / scale, center, scale


def _proper_axis_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        basis = np.eye(3)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = np.diag(signs) @ basis
            if np.linalg.det(candidate) > 0.5:
                rotations.append(candidate)
    return rotations


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    target_tree = cKDTree(target)
    probe = source[:: max(1, len(source) // 5000)]
    candidates = []
    for rotation in _proper_axis_rotations():
        transformed = probe @ rotation.T
        distances, _ = target_tree.query(transformed, k=1, workers=-1)
        candidates.append((float(np.mean(distances)), rotation))
    _, initial_rotation = min(candidates, key=lambda item: item[0])
    aligned = source @ initial_rotation.T
    iterations = []
    for _ in range(30):
        distances, indices = target_tree.query(aligned, k=1, workers=-1)
        cutoff = float(np.quantile(distances, 0.9))
        keep = distances <= cutoff
        rotation, translation = _kabsch(aligned[keep], target[indices[keep]])
        aligned = aligned @ rotation.T + translation
        mean = float(np.mean(distances[keep]))
        iterations.append(mean)
        if len(iterations) > 1 and abs(iterations[-2] - mean) < 1e-7:
            break
    return aligned, {
        "method": "24_axis_hypotheses_plus_trimmed_point_to_point_icp",
        "iterations": len(iterations),
        "mean_distance_history": [round(value, 8) for value in iterations],
    }


def _surface_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    target_tree = cKDTree(target)
    predicted_tree = cKDTree(predicted)
    pred_to_target, _ = target_tree.query(predicted, k=1, workers=-1)
    target_to_pred, _ = predicted_tree.query(target, k=1, workers=-1)
    metrics: dict[str, Any] = {
        "chamfer_l1_normalized": round(float(np.mean(pred_to_target) + np.mean(target_to_pred)), 8),
        "predicted_to_target_mean": round(float(np.mean(pred_to_target)), 8),
        "target_to_predicted_mean": round(float(np.mean(target_to_pred)), 8),
        "hausdorff_95_normalized": round(float(max(np.quantile(pred_to_target, .95), np.quantile(target_to_pred, .95))), 8),
    }
    f_scores = {}
    for threshold in (0.01, 0.02, 0.05):
        precision = float(np.mean(pred_to_target <= threshold))
        recall = float(np.mean(target_to_pred <= threshold))
        score = 2 * precision * recall / max(precision + recall, 1e-12)
        f_scores[f"{int(threshold * 100)}pct"] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f_score": round(score, 6),
        }
    metrics["surface_f_scores"] = f_scores
    return metrics


def _point_preview(predicted: np.ndarray, target: np.ndarray, path: Path) -> None:
    canvas = np.full((1500, 1500, 3), 245, dtype=np.uint8)
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    columns = (
        ((predicted, (175, 85, 40)), "IMAGE-ONLY PROPOSAL"),
        ((target, (40, 120, 185)), "WITHHELD OBJ"),
        (None, "ALIGNED OVERLAY"),
    )
    for row, (first_axis, second_axis, plane) in enumerate(projections):
        top = row * 500
        cv2.putText(canvas, f"{plane} PROJECTION", (12, top + 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (30, 30, 30), 2)
        for column, (single, title) in enumerate(columns):
            left = column * 500
            cv2.putText(canvas, title, (left + 65, top + 54), cv2.FONT_HERSHEY_SIMPLEX, .52, (30, 30, 30), 1)
            sources = (single,) if single else (
                (target, (40, 120, 185)),
                (predicted, (175, 85, 40)),
            )
            for points, color in sources:
                projected = (points[:, [first_axis, second_axis]] + .55) * 390
                projected[:, 1] = 390 - projected[:, 1]
                projected[:, 0] += left + 55
                projected[:, 1] += top + 78
                stride = max(1, len(projected) // 35000)
                for x, y in projected[::stride].astype(int):
                    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                        cv2.circle(canvas, (x, y), 1, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (0, top + 499), (1499, top + 499), (210, 210, 210), 1)
    cv2.imwrite(str(path), canvas)


def evaluate_against_withheld_obj(
    proposal_path: Path, reconstruction_record_path: Path, ground_truth_path: Path, output_dir: Path
) -> Path:
    record = json.loads(reconstruction_record_path.read_text(encoding="utf-8"))
    if record.get("ground_truth_accessed") is not False or record.get("ground_truth_path_recorded") is not False:
        raise ValueError("reconstruction record does not satisfy the blind-input contract")
    predicted_mesh = _load_mesh(proposal_path)
    target_mesh = _load_mesh(ground_truth_path)
    np.random.seed(17)
    predicted_points, _ = trimesh.sample.sample_surface(predicted_mesh, 50000)
    target_points, _ = trimesh.sample.sample_surface(target_mesh, 50000)
    predicted_normalized, _, _ = _normalize(predicted_points)
    target_normalized, _, _ = _normalize(target_points)
    predicted_aligned, alignment = _align(predicted_normalized, target_normalized)
    metrics = _surface_metrics(predicted_aligned, target_normalized)
    preview_path = output_dir / "aligned_front_comparison.png"
    _point_preview(predicted_aligned, target_normalized, preview_path)
    report = {
        "schema_version": "saraswati_blind_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "post_hoc_evaluation",
        "blind_contract": {
            "ground_truth_excluded_from_reconstruction": True,
            "reconstruction_record": str(reconstruction_record_path),
            "proposal_sha256": _sha256(proposal_path),
            "ground_truth_revealed_after_proposal": str(ground_truth_path.resolve()),
            "ground_truth_sha256": _sha256(ground_truth_path),
        },
        "sampling": {"proposal_points": 50000, "ground_truth_points": 50000, "seed": 17},
        "alignment": alignment,
        "surface_metrics": metrics,
        "preview": str(preview_path),
        "interpretation": {
            "measures_normalized_shape_similarity": True,
            "measures_metric_accuracy": False,
            "validates_manufacturing_accuracy": False,
            "expected_baseline": "coarse silhouette hull; fine relief is intentionally absent",
        },
    }
    report_path = output_dir / "evaluation_report.json"
    _write_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, default=DEFAULT_IMAGES["front"])
    parser.add_argument("--top", type=Path, default=DEFAULT_IMAGES["top"])
    parser.add_argument("--left", type=Path, default=DEFAULT_IMAGES["lsv"])
    parser.add_argument("--right", type=Path, default=DEFAULT_IMAGES["rsv"])
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    images = {"front": args.front, "top": args.top, "lsv": args.left, "rsv": args.right}
    proposal, record = reconstruct_from_images(images, args.output, args.resolution)
    report = evaluate_against_withheld_obj(proposal, record, args.ground_truth, args.output)
    print(report)


if __name__ == "__main__":
    main()
