"""Build reversible sculptural-relief evidence for all STL-1 ring views.

This stage deliberately does not emboss image brightness into the accepted CAD
body.  It combines retained depth residuals, image-detail support and depth
uncertainty, then writes a separate front-view OBJ hypothesis.  The OBJ uses
nominal display units and remains replaceable until cross-view or reviewed
evidence supports fusion into exact CAD.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "lalitha_phase5_stl1_relief_evidence_v1"
COMPLETE_METRIC_BACKENDS = (
    "depth_anything_v2_metric",
    "metric3d_v2",
    "unidepth_v2",
)
VIEWS = ("front", "top", "iso", "lsv", "rsv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _robust_scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[mask]
    if not selected.size:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(selected, (2.0, 98.0))
    if high <= low:
        high = low + 1e-6
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _strength(values: np.ndarray, mask: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=np.float32))
    selected = absolute[mask]
    scale = float(np.percentile(selected, percentile)) if selected.size else 1.0
    return np.clip(absolute / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def _align_depth_hypothesis(
    candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    normalized = _robust_scale(candidate, mask)
    if np.count_nonzero(mask) < 2:
        return normalized
    correlation = float(np.corrcoef(normalized[mask], reference[mask])[0, 1])
    if np.isfinite(correlation) and correlation < 0.0:
        normalized = 1.0 - normalized
    return normalized


def build_relief_evidence(
    *,
    depth: np.ndarray,
    uncertainty_u16: np.ndarray,
    gradient: np.ndarray,
    local_contrast: np.ndarray,
    jewelry_mask: np.ndarray,
    opening_mask: np.ndarray,
    metric_depths: tuple[np.ndarray, ...] = (),
) -> dict[str, np.ndarray | float | bool]:
    """Create a conservative relief proposal and expose rejected photo detail."""

    support = (jewelry_mask > 0) & ~(opening_mask > 0)
    support = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    base_depth = _robust_scale(np.asarray(depth, np.float32), support)

    aligned_metric = tuple(
        _align_depth_hypothesis(np.asarray(item, np.float32), base_depth, support)
        for item in metric_depths
    )
    if aligned_metric:
        stack = np.stack((base_depth, *aligned_metric))
        depth_hypothesis = np.median(stack, axis=0).astype(np.float32)
        disagreement = np.std(stack, axis=0).astype(np.float32)
        disagreement_strength = _strength(disagreement, support, 95.0)
        model_agreement = 1.0 - disagreement_strength
        multi_model = True
    else:
        depth_hypothesis = base_depth
        disagreement = np.zeros_like(base_depth)
        model_agreement = np.ones_like(base_depth)
        multi_model = False

    support_f32 = support.astype(np.float32)
    coarse_numerator = cv2.GaussianBlur(
        depth_hypothesis * support_f32, (0, 0), 12.0
    )
    coarse_denominator = cv2.GaussianBlur(support_f32, (0, 0), 12.0)
    coarse = coarse_numerator / np.maximum(coarse_denominator, 1e-6)
    signed_residual = (depth_hypothesis - coarse).astype(np.float32)
    depth_detail = _strength(signed_residual, support, 98.0)
    gradient_support = _strength(gradient, support, 98.0)
    contrast_support = _strength(local_contrast, support, 98.0)
    photo_detail = np.maximum(gradient_support, contrast_support)
    uncertainty = np.asarray(uncertainty_u16, np.float32) / 65535.0
    repeat_confidence = np.clip(1.0 - uncertainty, 0.0, 1.0)

    # Depth must contribute to confidence.  This prevents a bright highlight or
    # dark reflection from becoming relief solely because it has image contrast.
    boundary_distance = cv2.distanceTransform(
        support.astype(np.uint8), cv2.DIST_L2, 5
    )
    boundary_confidence = np.clip(boundary_distance / 10.0, 0.0, 1.0)
    confidence = (
        photo_detail
        * (0.15 + 0.85 * depth_detail)
        * repeat_confidence
        * model_agreement
        * boundary_confidence
    ).astype(np.float32)
    confidence[~support] = 0.0
    relief = (signed_residual * confidence).astype(np.float32)
    if np.any(support):
        relief[support] -= float(np.mean(relief[support]))
    relief[~support] = 0.0
    rejected = (photo_detail * (1.0 - depth_detail) * repeat_confidence).astype(
        np.float32
    )
    rejected[~support] = 0.0
    disagreement[~support] = 0.0

    detail_pixels = support & (photo_detail >= 0.25)
    supported_pixels = detail_pixels & (confidence >= 0.20)
    unresolved_pixels = detail_pixels & (rejected >= 0.35)
    detail_count = max(int(np.count_nonzero(detail_pixels)), 1)
    return {
        "support": support,
        "depth_hypothesis": depth_hypothesis,
        "signed_depth_residual": signed_residual,
        "photo_detail": photo_detail,
        "confidence": confidence,
        "relief": relief,
        "rejected_photo_detail": rejected,
        "model_disagreement": disagreement,
        "multi_model": multi_model,
        "mean_confidence": float(np.mean(confidence[support])) if np.any(support) else 0.0,
        "detail_supported_fraction": float(np.count_nonzero(supported_pixels) / detail_count),
        "detail_unresolved_fraction": float(np.count_nonzero(unresolved_pixels) / detail_count),
    }


def _color(values: np.ndarray, mask: np.ndarray, *, signed: bool = False) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if signed:
        selected = np.abs(values[mask])
        scale = float(np.percentile(selected, 98.0)) if selected.size else 1.0
        normalized = np.clip(values / max(scale, 1e-6) * 0.5 + 0.5, 0.0, 1.0)
        colored = cv2.applyColorMap(
            np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
    else:
        normalized = np.clip(values, 0.0, 1.0)
        colored = cv2.applyColorMap(
            np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )
    colored[~mask] = (238, 238, 238)
    return colored


def _panel(image: np.ndarray, title: str, width: int = 230) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((resized.shape[0] + 34, width, 3), 246, np.uint8)
    canvas[34:] = resized
    cv2.putText(canvas, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _write_review_sheet(
    source: np.ndarray,
    evidence: dict[str, Any],
    output: Path,
    title: str,
) -> None:
    mask = evidence["support"]
    source_panel = source.copy()
    source_panel[~mask] = (238, 238, 238)
    panels = (
        _panel(source_panel, "SOURCE"),
        _panel(_color(evidence["depth_hypothesis"], mask), "DEPTH HYPOTHESIS"),
        _panel(_color(evidence["photo_detail"], mask), "PHOTO DETAIL"),
        _panel(_color(evidence["confidence"], mask), "GEOMETRIC CONFIDENCE"),
        _panel(_color(evidence["relief"], mask, signed=True), "SIGNED RELIEF PROPOSAL"),
        _panel(_color(evidence["rejected_photo_detail"], mask), "REJECTED / UNRESOLVED"),
    )
    row = np.hstack(panels)
    header = np.full((58, row.shape[1], 3), 246, np.uint8)
    cv2.putText(header, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (27, 32, 37), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), np.vstack((header, row)))


def _write_relief_obj(
    output: Path,
    relief: np.ndarray,
    support: np.ndarray,
    width: float,
    height: float,
    grid_size: int = 96,
) -> tuple[int, int]:
    ys, xs = np.where(support)
    if not len(xs):
        raise ValueError("Cannot build relief OBJ from an empty support mask")
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    crop_relief = relief[y0:y1, x0:x1]
    crop_support = support[y0:y1, x0:x1].astype(np.uint8)
    target_w = grid_size
    target_h = max(8, int(round(grid_size * crop_relief.shape[0] / crop_relief.shape[1])))
    sampled_relief = cv2.resize(crop_relief, (target_w, target_h), interpolation=cv2.INTER_AREA)
    sampled_support = cv2.resize(crop_support, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
    selected = np.abs(sampled_relief[sampled_support])
    amplitude = float(np.percentile(selected, 98.0)) if selected.size else 1.0
    normalized = np.clip(sampled_relief / max(amplitude, 1e-6), -1.0, 1.0)
    z_scale = min(width, height) * 0.025

    indices = np.full((target_h, target_w), -1, np.int32)
    vertices: list[tuple[float, float, float]] = []
    for row in range(target_h):
        for column in range(target_w):
            if not sampled_support[row, column]:
                continue
            x = (column / max(target_w - 1, 1) - 0.5) * width
            y = (0.5 - row / max(target_h - 1, 1)) * height
            z = float(normalized[row, column] * z_scale)
            indices[row, column] = len(vertices) + 1
            vertices.append((x, y, z))
    faces: list[tuple[int, int, int]] = []
    for row in range(target_h - 1):
        for column in range(target_w - 1):
            a, b = indices[row, column], indices[row, column + 1]
            c, d = indices[row + 1, column], indices[row + 1, column + 1]
            if min(a, b, c, d) < 0:
                continue
            faces.extend(((a, c, b), (b, c, d)))
    lines = [
        "# Lalitha Phase 5.4 reversible relief evidence",
        "# Units are nominal display units, not millimetres.",
        "# This open surface is not an accepted solid or manufacturing model.",
    ]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(vertices), len(faces)


def _render_obj_preview(obj_path: Path, output_path: Path, title: str) -> None:
    """Render the written OBJ itself so the mesh artifact is visually audited."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = []
    faces = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(tuple(int(value) - 1 for value in line.split()[1:4]))
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int32)
    mesh = points[triangles]
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    xy_radius = max(float((high[:2] - low[:2]).max()) / 2.0, 0.5)
    cameras = (
        ("OBLIQUE", 32, -58),
        ("ORNAMENT FACE", 90, -90),
        ("RELIEF PROFILE", 0, -90),
    )
    figure = plt.figure(figsize=(12, 4.5), facecolor="#f4f4f4")
    for index, (label, elevation, azimuth) in enumerate(cameras, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        collection = Poly3DCollection(
            mesh,
            facecolor="#b7d3dc",
            edgecolor="#6d8790",
            linewidth=0.03,
            rasterized=True,
        )
        axis.add_collection3d(collection)
        axis.set_xlim(center[0] - xy_radius, center[0] + xy_radius)
        axis.set_ylim(center[1] - xy_radius, center[1] + xy_radius)
        axis.set_zlim(-xy_radius, xy_radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(label, loc="left", fontsize=10, fontweight="bold")
        axis.set_axis_off()
    figure.suptitle(title, fontsize=15, fontweight="bold", color="#20252b")
    figure.text(
        0.5,
        0.015,
        "Actual OBJ evidence surface - nominal units - not fused, solid or manufacturing-ready",
        ha="center",
        fontsize=9.5,
        color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.05, 0.99, 0.92))
    figure.savefig(output_path, dpi=155, bbox_inches="tight")
    plt.close(figure)


def _render_audit(root: Path, objects: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 4, figsize=(14, 19), facecolor="#f4f4f4")
    for axis, item in zip(axes.flat, objects):
        view = item["views"]["front"]
        source = cv2.imread(str(root / view["source_image"]), cv2.IMREAD_COLOR)
        confidence = cv2.imread(
            str(root / view["artifacts"]["confidence"]), cv2.IMREAD_COLOR
        )
        overlay = cv2.addWeighted(source, 0.58, confidence, 0.42, 0.0)
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        diagnostics = view["diagnostics"]
        holdout = " - HELD OUT" if item["split"] == "test" else ""
        axis.set_title(
            f"{item['object_id'].upper()} - {item['split'].upper()}{holdout}\n"
            f"supported {diagnostics['detail_supported_fraction']:.1%} - "
            f"unresolved {diagnostics['detail_unresolved_fraction']:.1%}",
            loc="left",
            fontsize=8.2,
            fontweight="bold",
        )
        axis.set_axis_off()
    figure.suptitle(
        "PHASE 5.4 - STL-1 RELIEF-EVIDENCE COVERAGE",
        fontsize=17,
        fontweight="bold",
        color="#20252b",
    )
    figure.text(
        0.5,
        0.012,
        "Colour overlay = depth-supported detail confidence; dark regions remain unresolved, not accepted relief",
        ha="center",
        fontsize=10,
        color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _metric_depth_records(root: Path) -> dict[str, tuple[Path, ...]]:
    experiment = root / "dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1"
    records: dict[str, list[Path]] = {}
    if not experiment.exists():
        return {}
    for backend in COMPLETE_METRIC_BACKENDS:
        report_path = experiment / "backends" / backend / "backend_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for record in report["records"]:
            records.setdefault(record["sample_key"], []).append(
                root / record["artifacts"]["depth_m"]
            )
    return {
        key: tuple(paths)
        for key, paths in records.items()
        if len(paths) == len(COMPLETE_METRIC_BACKENDS)
    }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record["diagnostics"][key] for record in records]
    return round(float(np.mean(values)), 6) if values else None


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    source_report_path = root / "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
    parameter_report_path = root / "data/phase5_stl1_ring_parameters_v1/report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    parameter_report = json.loads(parameter_report_path.read_text(encoding="utf-8"))
    parameters = {item["object_id"]: item for item in parameter_report["objects"]}
    metric_records = _metric_depth_records(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []

    for source_object in source_report["objects"]:
        object_id = source_object["object_id"]
        split = source_object["split"]
        object_dir = output_dir / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        views: dict[str, Any] = {}
        front_evidence: dict[str, Any] | None = None
        front_source: np.ndarray | None = None
        for view in VIEWS:
            source_view = source_object["views"][view]
            bundle_path = root / source_view["bundle"]
            with np.load(bundle_path) as bundle:
                source_bgr = cv2.cvtColor(bundle["normalized_rgb_u8"], cv2.COLOR_RGB2BGR)
                sample_key = f"{object_id}:{view}"
                metric_depths = tuple(
                    np.load(path) for path in metric_records.get(sample_key, ())
                )
                kwargs = {
                    "depth": bundle["relative_depth_regularized_f32"],
                    "uncertainty_u16": bundle["relative_depth_uncertainty_u16"],
                    "gradient": bundle["gradient_magnitude_f32"],
                    "local_contrast": bundle["local_contrast_f32"],
                    "jewelry_mask": bundle["mask_jewelry_u8"],
                    "opening_mask": bundle["mask_negative_space_proposal_u8"],
                    "metric_depths": metric_depths,
                }
                evidence = build_relief_evidence(**kwargs)
                repeated = build_relief_evidence(**kwargs)
            deterministic = all(
                np.array_equal(evidence[key], repeated[key])
                for key in ("confidence", "relief", "rejected_photo_detail")
            )
            paths = {
                "confidence": object_dir / f"{view}_relief_confidence.png",
                "relief": object_dir / f"{view}_signed_relief.png",
                "rejected": object_dir / f"{view}_rejected_photo_detail.png",
            }
            cv2.imwrite(str(paths["confidence"]), _color(evidence["confidence"], evidence["support"]))
            cv2.imwrite(str(paths["relief"]), _color(evidence["relief"], evidence["support"], signed=True))
            cv2.imwrite(str(paths["rejected"]), _color(evidence["rejected_photo_detail"], evidence["support"]))
            if split == "val" and view == "front":
                paths["review_sheet"] = object_dir / f"{object_id}_front_relief_review.png"
                _write_review_sheet(source_bgr, evidence, paths["review_sheet"], f"PHASE 5.4 - {object_id.upper()} RELIEF EVIDENCE")
            views[view] = {
                "source_image": source_view["normalized_source"],
                "source_bundle": source_view["bundle"],
                "multi_model_depth_available": bool(evidence["multi_model"]),
                "repeat_deterministic": deterministic,
                "diagnostics": {
                    "mean_confidence": round(float(evidence["mean_confidence"]), 6),
                    "detail_supported_fraction": round(float(evidence["detail_supported_fraction"]), 6),
                    "detail_unresolved_fraction": round(float(evidence["detail_unresolved_fraction"]), 6),
                },
                "artifacts": {key: _relative(path, root) for key, path in paths.items()},
            }
            if view == "front":
                front_evidence = evidence
                front_source = source_bgr

        assert front_evidence is not None and front_source is not None
        front_measurement = parameters[object_id]["views"]["front"]["measurement"]
        bbox_width, bbox_height = front_measurement["foreground_bbox_xywh"][2:4]
        # The Phase 5 parameter proposal intentionally has no physical head
        # dimensions.  Keep this evidence mesh scale-normalized instead of
        # inventing millimetres from the image.
        width = 1.0
        height = width * float(bbox_height) / max(float(bbox_width), 1.0)
        obj_path = object_dir / f"{object_id}_front_relief_evidence.obj"
        vertex_count, face_count = _write_relief_obj(
            obj_path,
            front_evidence["relief"],
            front_evidence["support"],
            width,
            height,
        )
        mesh_preview = None
        if split == "val":
            mesh_preview = object_dir / f"{object_id}_front_relief_obj_preview.png"
            _render_obj_preview(
                obj_path,
                mesh_preview,
                f"PHASE 5.4 - {object_id.upper()} RELIEF OBJ",
            )
        objects.append(
            {
                "object_id": object_id,
                "category": "ring",
                "split": split,
                "evaluation_status": "held_out_not_aggregated" if split == "test" else "evidence_diagnostics_only",
                "views": views,
                "front_relief_mesh": {
                    "path": _relative(obj_path, root),
                    "sha256": _sha256(obj_path),
                    "vertex_count": vertex_count,
                    "face_count": face_count,
                    "units": "nominal_display_units_not_metric",
                    "topology": "open_surface_not_solid",
                    "fusion_status": "separate_reversible_evidence_layer",
                    "preview": (
                        _relative(mesh_preview, root) if mesh_preview else None
                    ),
                },
            }
        )

    train_views = [view for item in objects if item["split"] == "train" for view in item["views"].values()]
    val_views = [view for item in objects if item["split"] == "val" for view in item["views"].values()]
    test = [item for item in objects if item["split"] == "test"]
    all_views = [view for item in objects for view in item["views"].values()]
    audit_path = output_dir / "stl1_relief_evidence_audit.png"
    _render_audit(root, objects, audit_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_profile": "images_only_ring_relief_evidence",
        "source": {
            "phase1_complete_features": _relative(source_report_path, root),
            "phase5_ring_parameters": _relative(parameter_report_path, root),
            "metric_depth_supplement": "dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1",
        },
        "summary": {
            "object_count": len(objects),
            "view_count": len(all_views),
            "front_relief_obj_count": len(objects),
            "multi_model_supported_view_count": sum(view["multi_model_depth_available"] for view in all_views),
            "train_mean_confidence": _mean(train_views, "mean_confidence"),
            "train_mean_detail_supported_fraction": _mean(train_views, "detail_supported_fraction"),
            "validation_mean_confidence": _mean(val_views, "mean_confidence"),
            "validation_mean_detail_supported_fraction": _mean(val_views, "detail_supported_fraction"),
            "validation_mean_detail_unresolved_fraction": _mean(val_views, "detail_unresolved_fraction"),
            "test_held_out_count": len(test),
        },
        "checks": {
            "all_24_ring_objects_processed": len(objects) == 24,
            "all_120_views_processed": len(all_views) == 120,
            "all_front_relief_objs_present": all((root / item["front_relief_mesh"]["path"]).exists() for item in objects),
            "all_outputs_repeat_deterministic": all(view["repeat_deterministic"] for view in all_views),
            "all_meshes_nonempty": all(item["front_relief_mesh"]["vertex_count"] > 0 and item["front_relief_mesh"]["face_count"] > 0 for item in objects),
            "held_out_test_not_aggregated": len(test) == 3,
            "relief_kept_separate_from_exact_cad": True,
            "ring_only_scope_enforced": all(item["category"] == "ring" for item in objects),
        },
        "objects": objects,
        "artifacts": {"audit": _relative(audit_path, root)},
        "decision": "relief_evidence_ready_for_cross_view_review_not_cad_fusion",
        "claims": {
            "per_view_relief_evidence_available": True,
            "raw_brightness_embossed_directly": False,
            "relief_mesh_fused_into_exact_cad": False,
            "relief_accuracy_measured": False,
            "metric_scale_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The dataset provides images only; no relief, scan or physical ground truth exists.",
            "Depth Anything preserves broad shape but omits much of the fine sculptural carving visible in STL-1.",
            "Only 15 of 120 views have the retained three-model metric-depth supplement; other views use repeat uncertainty from Phase 1.",
            "The OBJ files are open evidence surfaces in nominal display units, not watertight solids or manufacturing geometry.",
            "Highlights, shadows and polished-metal reflections remain unresolved wherever depth does not support the image detail.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/phase5_stl1_relief_evidence_v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps({"report": str(output / "report.json"), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
