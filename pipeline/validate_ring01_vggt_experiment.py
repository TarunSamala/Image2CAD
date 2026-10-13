"""Validate and visualize the bounded Ring01 VGGT low-VRAM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh


VIEWS = ("front", "side", "top", "angled", "back")


def _square_image(path: Path, size: int, mask: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    side = max(height, width)
    shape = (side, side) if mask else (side, side, 3)
    canvas = np.zeros(shape, dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = image
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def _colour(values: np.ndarray, valid: np.ndarray, near_is_low: bool) -> np.ndarray:
    samples = values[valid]
    low, high = np.percentile(samples, (2, 98))
    span = max(float(high - low), 1e-8)
    normalized = np.clip((values - low) / span, 0.0, 1.0)
    if near_is_low:
        normalized = 1.0 - normalized
    colour = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    colour[~valid] = (245, 245, 245)
    return colour


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    header = np.full((54, image.shape[1], 3), 248, dtype=np.uint8)
    cv2.putText(header, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(header, subtitle, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (75, 75, 75), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def run(root: Path, run_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "vggt_report.json").read_text(encoding="utf-8"))
    cameras = np.load(run_dir / "vggt_cameras.npz")
    dense = np.load(run_dir / "vggt_dense_evidence.npz")
    extrinsic = cameras["extrinsic"][0]
    intrinsic = cameras["intrinsic"][0]
    depth = dense["depth"][0, ..., 0]
    depth_conf = dense["depth_conf"][0]
    world_points = dense["world_points"][0]
    world_conf = dense["world_points_conf"][0]
    size = int(report["image_size"])

    arrays = (extrinsic, intrinsic, depth, depth_conf, world_points, world_conf)
    checks = {
        "run_completed": report["status"] == "completed_machine_hypothesis",
        "five_views_present": depth.shape[0] == len(VIEWS),
        "all_arrays_finite": all(bool(np.isfinite(array).all()) for array in arrays),
        "camera_rotation_determinants_valid": all(
            abs(float(np.linalg.det(matrix[:, :3])) - 1.0) < 0.01 for matrix in extrinsic
        ),
        "intrinsics_positive": bool(np.all(intrinsic[:, 0, 0] > 0) and np.all(intrinsic[:, 1, 1] > 0)),
        "dense_shapes_valid": (
            depth.shape == (5, size, size)
            and world_points.shape == (5, size, size, 3)
        ),
        "noncommercial_research_only": (
            report["checkpoint_license_id"] == "VGGT-1B-NonCommercial"
            and report["decision"] == "research_only"
        ),
        "accuracy_not_claimed": not any(report["claims"].values()),
    }

    manifest = json.loads(
        (root / "data/ring01_ground_truth_v1/manifest.json").read_text(encoding="utf-8")
    )
    rows = []
    view_metrics = []
    point_sets = []
    colour_sets = []
    for index, view in enumerate(VIEWS):
        source_path = root / manifest["views"][view]["source"]["path"]
        mask_path = root / manifest["views"][view]["proposals"]["jewelry"]["path"]
        source = _square_image(source_path, size)
        valid = _square_image(mask_path, size, mask=True) > 0
        if not np.any(valid):
            raise ValueError(f"Empty jewellery proposal for {view}")
        point_norm = np.linalg.norm(world_points[index], axis=-1)
        point_sets.append(world_points[index][valid])
        colour_sets.append(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)[valid])
        depth_cell = _colour(depth[index], valid, near_is_low=True)
        depth_conf_cell = _colour(depth_conf[index], valid, near_is_low=False)
        point_cell = _colour(point_norm, valid, near_is_low=True)
        row = np.hstack(
            (
                _label(source, f"{view.upper()} SOURCE", "official 224 px preprocessing"),
                _label(depth_cell, "VGGT DEPTH", "red closer / blue farther"),
                _label(depth_conf_cell, "DEPTH CONFIDENCE", "red higher / blue lower"),
                _label(point_cell, "WORLD-POINT RADIUS", "relative, non-metric"),
            )
        )
        rows.append(row)
        values = depth[index][valid]
        view_metrics.append(
            {
                "view": view,
                "masked_pixel_count": int(valid.sum()),
                "depth_p02": round(float(np.percentile(values, 2)), 6),
                "depth_median": round(float(np.median(values)), 6),
                "depth_p98": round(float(np.percentile(values, 98)), 6),
                "depth_confidence_median": round(float(np.median(depth_conf[index][valid])), 6),
                "world_point_confidence_median": round(float(np.median(world_conf[index][valid])), 6),
                "rotation_determinant": round(float(np.linalg.det(extrinsic[index, :, :3])), 6),
            }
        )

    preview = np.vstack(rows)
    preview_path = run_dir / "ring01_vggt_dense_review.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Could not write {preview_path}")

    fused_points = np.concatenate(point_sets, axis=0)
    fused_colours = np.concatenate(colour_sets, axis=0)
    cloud_path = run_dir / "ring01_vggt_fused_points.ply"
    trimesh.points.PointCloud(fused_points, colors=fused_colours).export(cloud_path)

    sample_count = min(30_000, len(fused_points))
    indices = np.linspace(0, len(fused_points) - 1, sample_count, dtype=np.int64)
    sampled = fused_points[indices]
    sampled_colours = fused_colours[indices].astype(np.float32) / 255.0
    bounds = np.percentile(sampled, (1, 99), axis=0)
    centre = bounds.mean(axis=0)
    radius = max(float(np.max(bounds[1] - bounds[0])) / 2.0, 1e-6)
    figure = plt.figure(figsize=(10, 10), dpi=160, facecolor="white")
    camera_views = (
        ("ISOMETRIC", 25, -45),
        ("AXIS +Y", 0, -90),
        ("AXIS +X", 0, 0),
        ("TOP", 90, -90),
    )
    for panel, (title, elevation, azimuth) in enumerate(camera_views, start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        axis.scatter(
            sampled[:, 0], sampled[:, 1], sampled[:, 2],
            c=sampled_colours, s=0.7, linewidths=0, depthshade=False,
        )
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_xlim(centre[0] - radius, centre[0] + radius)
        axis.set_ylim(centre[1] - radius, centre[1] + radius)
        axis.set_zlim(centre[2] - radius, centre[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_axis_off()
    figure.suptitle("Ring01 VGGT fused world-point hypothesis · non-metric", fontsize=13)
    figure.tight_layout()
    cloud_preview_path = run_dir / "ring01_vggt_fused_points_review.png"
    figure.savefig(cloud_preview_path, bbox_inches="tight")
    plt.close(figure)

    result = {
        "schema_version": "ring01_vggt_low_vram_validation_v1",
        "source_report": str((run_dir / "vggt_report.json").relative_to(root)),
        "experiment": {
            "image_count": report["image_count"],
            "image_size": report["image_size"],
            "dtype": report["dtype"],
            "peak_cuda_bytes": report["peak_cuda_bytes"],
            "elapsed_seconds": report["elapsed_seconds"],
        },
        "checks": checks,
        "view_metrics": view_metrics,
        "fused_point_cloud": {
            "point_count": int(len(fused_points)),
            "bounds_min": [round(float(value), 6) for value in fused_points.min(axis=0)],
            "bounds_max": [round(float(value), 6) for value in fused_points.max(axis=0)],
            "units": "model_relative_non_metric",
        },
        "artifacts": {
            "review_preview": str(preview_path.relative_to(root)),
            "fused_point_cloud": str(cloud_path.relative_to(root)),
            "fused_point_cloud_preview": str(cloud_preview_path.relative_to(root)),
        },
        "decision": "ready_for_visual_review" if all(checks.values()) else "failed_machine_validation",
        "claims": {
            "camera_pose_accuracy_validated": False,
            "metric_depth_validated": False,
            "cross_view_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The 224 px low-VRAM run is below the official 518 px preprocessing path.",
            "Input views are independently framed product renders without calibrated cameras.",
            "Jewellery masks are unreviewed machine proposals used only to limit visual statistics.",
            "Finite matrices and valid rotations prove numerical integrity, not camera accuracy.",
        ],
    }
    (run_dir / "validation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/ring01_vggt_dense_224_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    result = run(root, run_dir)
    print(json.dumps({"decision": result["decision"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
