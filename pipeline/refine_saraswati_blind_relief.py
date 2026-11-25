"""Apply bounded front-depth relief to the blind Saraswati visual hull.

This proposal stage reads source images, masks and a precomputed depth
hypothesis. It never reads the withheld OBJ. Evaluation remains a separate
post-hoc call through ``evaluate_against_withheld_obj``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from reconstruct_dataset_phase3 import _fit_volume, _mesh_from_volume
from run_saraswati_blind_benchmark import (
    DEFAULT_GROUND_TRUTH,
    DEFAULT_IMAGES,
    _foreground_mask,
    _sha256,
    _write_json,
    evaluate_against_withheld_obj,
)


DEFAULT_BASELINE = Path("dataset/phase_runs/v1/saraswati_blind_benchmark_256_v1")
DEFAULT_OUTPUT = Path("dataset/phase_runs/v1/saraswati_blind_relief_v1")


def _warp_centered_values(
    values: np.ndarray, mask: np.ndarray, scale: float, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise ValueError("empty front mask")
    x, y, width, height = cv2.boundingRect(points)
    center_x, center_y = x + (width - 1) / 2, y + (height - 1) / 2
    matrix = np.asarray(
        [
            [scale, 0.0, (resolution - 1) / 2 - scale * center_x],
            [0.0, scale, (resolution - 1) / 2 - scale * center_y],
        ],
        dtype=np.float64,
    )
    warped_values = cv2.warpAffine(
        values.astype(np.float32), matrix, (resolution, resolution),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_mask = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (resolution, resolution),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    return warped_values, warped_mask


def _normalized_depth(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    selected = depth[mask]
    low, high = np.percentile(selected, (2, 98))
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((depth - low) / (high - low), 0, 1).astype(np.float32)
    normalized = cv2.bilateralFilter(normalized, 7, 0.08, 5)
    return normalized, {"p02_m": float(low), "p98_m": float(high)}


def carve_front_relief(
    volume: np.ndarray, normalized_depth: np.ndarray, valid: np.ndarray, strength: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0 < strength <= 0.2:
        raise ValueError("bounded relief strength must be in (0, 0.2]")
    carved = volume.copy()
    removed = 0
    affected = 0
    for z, x in zip(*np.where(valid)):
        occupied = np.flatnonzero(carved[z, :, x])
        if occupied.size < 3:
            continue
        front = int(occupied[0])
        thickness = int(occupied[-1] - occupied[0] + 1)
        inset = int(round(float(normalized_depth[z, x]) * strength * thickness))
        inset = min(inset, max(0, thickness - 2))
        if inset:
            carved[z, front:front + inset, x] = False
            removed += inset
            affected += 1
    if not np.any(carved):
        raise RuntimeError("relief carving removed the entire volume")
    return carved, {
        "strength_fraction_of_local_visual_hull_thickness": strength,
        "removed_voxels": removed,
        "affected_front_columns": affected,
        "remaining_fraction": round(float(np.count_nonzero(carved) / np.count_nonzero(volume)), 7),
    }


def refine(
    baseline_root: Path, output_dir: Path, strength: float = 0.08, resolution: int = 256
) -> tuple[Path, Path]:
    depth_audit = json.loads((baseline_root / "depth_audit.json").read_text(encoding="utf-8"))
    if depth_audit["ground_truth_obj_accessed"] is not False:
        raise ValueError("depth evidence does not satisfy blind-input contract")
    if not depth_audit["views"]["front"]["eligible_as_relative_relief_hypothesis"]:
        raise ValueError("front depth failed the source-image edge agreement gate")

    images = DEFAULT_IMAGES
    masks: dict[str, np.ndarray] = {}
    for view, path in images.items():
        masks[view], _ = _foreground_mask(path)
    volume, side_mode, fit, _ = _fit_volume(masks, resolution)

    backend_report = json.loads(
        (baseline_root / "depth_backends/depth_anything_v2_metric/backend_report.json").read_text(encoding="utf-8")
    )
    front_record = next(record for record in backend_report["records"] if record["view"] == "front")
    depth_path = Path(front_record["artifacts"]["depth_m"])
    depth = np.load(depth_path)
    normalized, depth_range = _normalized_depth(depth, masks["front"])
    scale = float(fit["axis_alignment"]["warp_scales_input_px_to_voxel"]["front"])
    warped_depth, warped_valid = _warp_centered_values(normalized, masks["front"], scale, resolution)
    refined_volume, carve = carve_front_relief(volume, warped_depth, warped_valid, strength)
    mesh = _mesh_from_volume(refined_volume, nominal_size_mm=50.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "saraswati_image_depth_relief_proposal.obj"
    mesh.export(mesh_path)
    depth_preview = cv2.applyColorMap(np.round((1 - warped_depth) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_preview[~warped_valid] = 238
    cv2.imwrite(str(output_dir / "front_depth_warped.png"), depth_preview)
    record = {
        "schema_version": "saraswati_blind_relief_reconstruction_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "reconstruction_refinement",
        "input_contract": "four_images_plus_depth_hypothesis_no_ground_truth_mesh",
        "ground_truth_accessed": False,
        "ground_truth_path_recorded": False,
        "baseline_reconstruction_record": str(baseline_root / "reconstruction_record.json"),
        "depth_audit": str(baseline_root / "depth_audit.json"),
        "depth_artifact": {"path": str(depth_path), "sha256": _sha256(depth_path), **depth_range},
        "method": "bounded_front_surface_relative_depth_carve",
        "side_candidate": side_mode,
        "carve": carve,
        "proposal": {
            "path": str(mesh_path),
            "sha256": _sha256(mesh_path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
        },
        "claims": {
            "fine_relief_recovered": False,
            "metric_scale_recovered": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    record_path = output_dir / "reconstruction_record.json"
    _write_json(record_path, record)
    return mesh_path, record_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    args = parser.parse_args()
    proposal, record = refine(args.baseline, args.output, args.strength, args.resolution)
    report = evaluate_against_withheld_obj(proposal, record, args.ground_truth, args.output)
    print(report)


if __name__ == "__main__":
    main()
