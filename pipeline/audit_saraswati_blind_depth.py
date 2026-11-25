"""Audit blind Saraswati depth hypotheses without reading the withheld OBJ."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_ROOT = Path("dataset/phase_runs/v1/saraswati_blind_benchmark_256_v1")


def _normalize(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = depth[mask]
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        high = low + 1e-6
    return np.clip((depth - low) / (high - low), 0, 1).astype(np.float32), float(low), float(high)


def _edge_agreement(image: np.ndarray, normalized: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(gray, 70, 150) > 0
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    valid_values = magnitude[mask]
    threshold = float(np.percentile(valid_values, 80)) if valid_values.size else 0.0
    depth_edges = magnitude >= threshold
    dilated = cv2.dilate(image_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    selected = depth_edges & mask
    return float(np.mean(dilated[selected])) if np.any(selected) else 0.0


def _panel(title: str, image: np.ndarray, width: int = 350, height: int = 280) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 36, width, 3), 245, np.uint8)
    canvas[36:] = resized
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (24, 24, 28), 1, cv2.LINE_AA)
    return canvas


def audit(root: Path) -> Path:
    backend_report = root / "depth_backends/depth_anything_v2_metric/backend_report.json"
    report = json.loads(backend_report.read_text(encoding="utf-8"))
    rows = []
    records: dict[str, Any] = {}
    normalized_depths: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for record in report["records"]:
        view = record["view"]
        manifest = json.loads((root / "metric_depth_input_manifest.json").read_text(encoding="utf-8"))
        sample = next(item for item in manifest["samples"] if item["view"] == view)
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
        depth = np.load(record["artifacts"]["depth_m"])
        normalized, low, high = _normalize(depth, mask)
        normalized_depths[view] = normalized
        masks[view] = mask
        near_is_red = cv2.applyColorMap(np.round((1 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        near_is_red[~mask] = 238
        gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        gradient /= max(float(np.percentile(gradient[mask], 98)), 1e-6)
        gradient_color = cv2.applyColorMap(np.round(np.clip(gradient, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        gradient_color[~mask] = 238
        agreement = _edge_agreement(image, normalized, mask)
        records[view] = {
            "robust_depth_range_m": round(high - low, 7),
            "robust_depth_range_over_median": round((high - low) / max(float(np.median(depth[mask])), 1e-9), 7),
            "depth_gradient_to_image_edge_agreement": round(agreement, 6),
            "eligible_as_relative_relief_hypothesis": agreement >= 0.35,
        }
        rows.append(cv2.hconcat((
            _panel(f"{view.upper()} SOURCE", image),
            _panel("NORMALIZED DEPTH - RED NEAR", near_is_red),
            _panel("DEPTH GRADIENT", gradient_color),
        )))

    left = normalized_depths["lsv"]
    right = np.fliplr(normalized_depths["rsv"])
    common_shape = (min(left.shape[1], right.shape[1]), min(left.shape[0], right.shape[0]))
    left = cv2.resize(left, common_shape, interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, common_shape, interpolation=cv2.INTER_LINEAR)
    left_mask = cv2.resize(masks["lsv"].astype(np.uint8), common_shape, interpolation=cv2.INTER_NEAREST) > 0
    right_mask = cv2.resize(np.fliplr(masks["rsv"]).astype(np.uint8), common_shape, interpolation=cv2.INTER_NEAREST) > 0
    common = left_mask & right_mask
    correlation = float(np.corrcoef(left[common], right[common])[0, 1]) if np.count_nonzero(common) > 20 else 0.0

    sheet = cv2.vconcat(rows)
    sheet_path = root / "depth_audit.png"
    cv2.imwrite(str(sheet_path), sheet)
    result = {
        "schema_version": "saraswati_blind_depth_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_obj_accessed": False,
        "backend_report": str(backend_report),
        "views": records,
        "cross_view": {
            "left_right_mirrored_depth_correlation": round(correlation, 6),
            "camera_calibrated": False,
            "absolute_depths_directly_fusible": False,
        },
        "decision": (
            "relative_relief_experiment_allowed_with_bounded_displacement"
            if correlation >= 0.25 and sum(item["eligible_as_relative_relief_hypothesis"] for item in records.values()) >= 3
            else "reject_depth_fusion_pending_better_cross_view_consistency"
        ),
        "limits": [
            "Per-view robust normalization removes incompatible camera-distance offsets.",
            "No physical scale, calibrated cameras or jewellery-specific metric-depth validation is available.",
            "The withheld Saraswati OBJ was not read by this audit.",
        ],
        "preview": str(sheet_path),
    }
    result_path = root / "depth_audit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(audit(args.root))


if __name__ == "__main__":
    main()
