"""Validate STL-1 image/OBJ pairing through normalized silhouette agreement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from pipeline.run_saraswati_blind_benchmark import _proper_axis_rotations


PRINCIPAL_VIEWS = ("front", "top", "lsv", "rsv")


def _manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_points(path: Path, maximum: int = 80_000) -> np.ndarray:
    loaded = trimesh.load(path, force="scene", process=False)
    arrays = [
        np.asarray(mesh.vertices, dtype=np.float64)
        for mesh in loaded.geometry.values()
        if len(mesh.vertices)
    ]
    if not arrays:
        raise ValueError(f"no vertices in {path}")
    points = np.vstack(arrays)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
        points = points[indices]
    points -= np.median(points, axis=0)
    covariance = np.cov(points.T)
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return points @ basis


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise ValueError("empty reference mask")
    return cv2.boundingRect(points)


def _project(points: np.ndarray, view: str) -> np.ndarray:
    if view == "front":
        return points[:, [0, 2]] * np.asarray((1.0, -1.0))
    if view == "top":
        return points[:, [0, 1]] * np.asarray((1.0, -1.0))
    if view == "lsv":
        return points[:, [1, 2]] * np.asarray((1.0, -1.0))
    if view == "rsv":
        return points[:, [1, 2]] * np.asarray((-1.0, -1.0))
    raise ValueError(view)


def _rasterize(projected: np.ndarray, reference: np.ndarray) -> np.ndarray:
    height, width = reference.shape
    x, y, target_width, target_height = _bbox(reference)
    low = np.percentile(projected, 0.15, axis=0)
    high = np.percentile(projected, 99.85, axis=0)
    span = np.maximum(high - low, 1e-9)
    normalized = (projected - low) / span
    pixels = np.empty_like(normalized)
    pixels[:, 0] = x + normalized[:, 0] * max(1, target_width - 1)
    pixels[:, 1] = y + normalized[:, 1] * max(1, target_height - 1)
    pixels = np.round(pixels).astype(np.int32)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    mask = np.zeros_like(reference, dtype=np.uint8)
    mask[pixels[valid, 1], pixels[valid, 0]] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return mask > 0


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 0.0


def _panel(title: str, image: np.ndarray, size: int = 256) -> np.ndarray:
    value = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    if value.ndim == 2:
        value = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canvas = np.full((size + 30, size, 3), 245, np.uint8)
    canvas[30:] = value
    cv2.putText(canvas, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .44, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _audit_sheet(
    references: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    metrics: dict[str, float],
) -> np.ndarray:
    rows = []
    for view in PRINCIPAL_VIEWS:
        reference = references[view]
        proposal = projections[view]
        overlay = np.full((*reference.shape, 3), 245, np.uint8)
        overlay[reference] = (45, 145, 220)
        overlay[proposal] = (210, 105, 40)
        overlay[reference & proposal] = (75, 185, 95)
        rows.append(cv2.hconcat((
            _panel(f"{view.upper()} IMAGE MASK", reference.astype(np.uint8) * 255),
            _panel("OBJ PROJECTION", proposal.astype(np.uint8) * 255),
            _panel(f"OVERLAY IoU {metrics[view]:.3f}", overlay),
        )))
    return cv2.vconcat(rows)


def validate(
    dataset_root: Path,
    manifest_path: Path,
    output_root: Path,
    size: int = 256,
) -> dict[str, Any]:
    records = _manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    objects = []
    rotations = _proper_axis_rotations()
    for record in records:
        directory = dataset_root / record["source_object_name"]
        obj_files = list(directory.glob("*.obj"))
        if len(obj_files) != 1:
            raise ValueError(f"expected one OBJ in {directory}")
        references = {}
        for view in PRINCIPAL_VIEWS:
            mask = cv2.imread(record["views"][view]["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(record["views"][view]["jewelry_mask_path"])
            references[view] = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 127

        points = _load_points(obj_files[0])
        candidates = []
        for index, rotation in enumerate(rotations):
            oriented = points @ rotation.T
            projections = {
                view: _rasterize(_project(oriented, view), references[view])
                for view in PRINCIPAL_VIEWS
            }
            metrics = {
                view: _iou(references[view], projections[view])
                for view in PRINCIPAL_VIEWS
            }
            candidates.append((float(np.mean(list(metrics.values()))), index, metrics, projections))
        score, rotation_index, metrics, projections = max(candidates, key=lambda item: item[0])
        object_root = output_root / record["object_id"]
        object_root.mkdir(exist_ok=True)
        audit_path = object_root / "pairing_audit.png"
        cv2.imwrite(str(audit_path), _audit_sheet(references, projections, metrics))
        cross_view_status = (
            "strong" if score >= 0.70 else "review" if score >= 0.50 else "weak"
        )
        objects.append({
            "object_id": record["object_id"],
            "split": record["split"],
            "mesh_path": str(obj_files[0]),
            "rotation_hypothesis": rotation_index,
            "mean_principal_iou": round(score, 6),
            "view_iou": {view: round(metrics[view], 6) for view in PRINCIPAL_VIEWS},
            "audit": str(audit_path),
            "pairing_status": (
                "front_supported" if metrics["front"] >= 0.70 else "manual_review"
            ),
            "cross_view_geometry_status": cross_view_status,
        })

    scores = [item["mean_principal_iou"] for item in objects]
    report = {
        "schema_version": "image2cad_stl1_mesh_image_pairing_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "PCA_plus_24_axis_hypotheses_normalized_orthographic_point_projection",
        "object_count": len(objects),
        "mean_iou": round(float(np.mean(scores)), 6),
        "median_iou": round(float(np.median(scores)), 6),
        "minimum_iou": round(float(np.min(scores)), 6),
        "pairing_status_counts": {
            status: sum(item["pairing_status"] == status for item in objects)
            for status in ("front_supported", "manual_review")
        },
        "cross_view_geometry_status_counts": {
            status: sum(item["cross_view_geometry_status"] == status for item in objects)
            for status in ("strong", "review", "weak")
        },
        "claim_policy": {
            "validates_pairing": "image-space screening only",
            "validates_3d_accuracy": False,
            "validates_manufacturing_accuracy": False,
        },
        "objects": objects,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/STL-1"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/prepared_v1/manifest.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/phase_runs/v1/stl1_mesh_image_pairing_v1"),
    )
    args = parser.parse_args()
    report = validate(args.dataset, args.manifest, args.output)
    print(json.dumps({key: report[key] for key in (
        "object_count", "mean_iou", "median_iou", "minimum_iou",
        "pairing_status_counts", "cross_view_geometry_status_counts"
    )}, indent=2))


if __name__ == "__main__":
    main()
