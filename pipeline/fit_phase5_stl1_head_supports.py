"""Fit exact ring head-support envelopes from STL-1 multi-view silhouettes.

The checkpoint adds a front-profile head blank and two shoulder supports to the
Phase 5.1 exact shank.  It models only the load-bearing envelope: relief,
openwork, stones, prongs and under-gallery cavities remain separate future work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.exact_cad_v2 import (
    HeadSupportSpec,
    RingBodySpec,
    build_head_support_envelope,
    build_profiled_ring_body,
)
from pipeline.fit_phase5_stl1_ring_bodies import (
    NOMINAL_INNER_DIAMETER,
    _boundary_f1,
    _iou,
    _profile_controls,
)


SCHEMA_VERSION = "lalitha_phase5_stl1_exact_head_support_fit_v1"


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No silhouette contour")
    return max(contours, key=cv2.contourArea)[:, 0, :]


def _sample_closed_contour(contour: np.ndarray, count: int = 96) -> np.ndarray:
    points = contour.astype(np.float64)
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total <= 0.0:
        raise ValueError("Degenerate silhouette contour")
    samples = []
    for distance in np.linspace(0.0, total, count, endpoint=False):
        segment = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
        fraction = (distance - cumulative[segment]) / max(lengths[segment], 1e-9)
        samples.append(closed[segment] * (1.0 - fraction) + closed[segment + 1] * fraction)
    return np.asarray(samples)


def _bulky_side_region(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = _bbox(mask)
    column_heights = np.count_nonzero(mask, axis=0)
    threshold = max(3, int(column_heights.max() * 0.32))
    columns = np.where(column_heights >= threshold)[0]
    if not len(columns):
        return mask.copy(), (x0, y0, x1, y1)
    head_x0, head_x1 = int(columns.min()), int(columns.max() + 1)
    region = np.zeros_like(mask)
    region[:, head_x0:head_x1] = mask[:, head_x0:head_x1]
    rx0, ry0, rx1, ry1 = _bbox(region)
    return region, (rx0, ry0, rx1, ry1)


def _head_dimensions(
    source: dict[str, Any],
    root: Path,
    ring_body_spec: RingBodySpec,
) -> dict[str, float]:
    evidence = source["ring_parameter_evidence"]
    aperture = evidence["aperture"]
    diameter_px = float(aperture["equivalent_diameter_px"])
    cx, cy = aperture["center_xy_px"]
    top_mask = cv2.imread(
        str(root / source["views"]["top"]["mask_path"]), cv2.IMREAD_GRAYSCALE
    ) > 127
    start_y = int(round(cy + float(aperture["minor_axis_px"]) * 0.32))
    lower = np.zeros_like(top_mask)
    lower[max(0, start_y):] = top_mask[max(0, start_y):]
    x0, y0, x1, y1 = _bbox(lower)
    units_per_pixel = NOMINAL_INNER_DIAMETER / diameter_px
    head_width = max((x1 - x0) * units_per_pixel, ring_body_spec.radial_thickness * 2.5)
    head_height = max((y1 - start_y) * units_per_pixel, ring_body_spec.radial_thickness * 2.0)

    front_mask = cv2.imread(
        str(root / source["views"]["front"]["mask_path"]), cv2.IMREAD_GRAYSCALE
    ) > 127
    fx0, fy0, fx1, fy1 = _bbox(front_mask)
    front_depth = head_width * (fy1 - fy0) / max(fx1 - fx0, 1)
    side_depths = []
    for view in ("lsv", "rsv"):
        side_mask = cv2.imread(
            str(root / source["views"][view]["mask_path"]), cv2.IMREAD_GRAYSCALE
        ) > 127
        _, (sx0, sy0, sx1, sy1) = _bulky_side_region(side_mask)
        # Side views project Z horizontally and Y vertically.  Convert their
        # pixel aspect into physical Y using the top-view Z extent.
        side_depths.append(head_height * (sy1 - sy0) / max(sx1 - sx0, 1))
    # Preserve the direct ornament-face aspect ratio.  The side views are kept
    # as diagnostics because their independent framing is not calibrated well
    # enough to override the direct X-Y silhouette.
    head_depth = float(np.clip(
        front_depth,
        ring_body_spec.axial_depth * 1.5,
        head_width * 1.35,
    ))
    side_depth_median = float(np.median(side_depths))
    return {
        "head_width": float(head_width),
        "head_depth": head_depth,
        "head_height": float(head_height),
        "front_depth_estimate": float(front_depth),
        "side_depth_estimates": [float(value) for value in side_depths],
        "side_depth_median": side_depth_median,
        "front_side_depth_disagreement_fraction": float(
            abs(front_depth - side_depth_median) / max(front_depth, 1e-9)
        ),
        "top_start_y_px": float(start_y),
        "top_center_x_px": float((x0 + x1) / 2.0),
        "top_bbox_x0_px": float(x0),
        "top_bbox_x1_px": float(x1),
    }


def _front_profile(
    mask: np.ndarray, width: float, depth: float
) -> tuple[tuple[float, float], ...]:
    contour = _sample_closed_contour(_largest_contour(mask), 96)
    x0, y0, x1, y1 = _bbox(mask)
    center_x = (x0 + x1 - 1) / 2.0
    center_y = (y0 + y1 - 1) / 2.0
    scale_x = width / max(x1 - x0, 1)
    scale_y = depth / max(y1 - y0, 1)
    return tuple(
        ((float(x) - center_x) * scale_x, (float(y) - center_y) * scale_y)
        for x, y in contour
    )


def _render_front_profile(
    profile: tuple[tuple[float, float], ...],
    mask_shape: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    width: float,
    depth: float,
) -> np.ndarray:
    x0, y0, x1, y1 = target_bbox
    center_x = (x0 + x1 - 1) / 2.0
    center_y = (y0 + y1 - 1) / 2.0
    scale_x = (x1 - x0) / width
    scale_y = (y1 - y0) / depth
    polygon = np.rint(np.asarray([
        (center_x + x * scale_x, center_y + y * scale_y) for x, y in profile
    ])).astype(np.int32)
    rendered = np.zeros(mask_shape, np.uint8)
    cv2.fillPoly(rendered, [polygon], 255)
    return rendered


def _top_profile(
    mask: np.ndarray,
    center_x_px: float,
    start_y_px: int,
    units_per_pixel: float,
    base_z: float,
) -> tuple[tuple[float, float], ...]:
    """Convert the observed top-view support contour into CAD X-Z controls."""
    contour = _sample_closed_contour(_largest_contour(mask), 96)
    return tuple(
        (
            (float(x) - center_x_px) * units_per_pixel,
            base_z + (float(y) - start_y_px) * units_per_pixel,
        )
        for x, y in contour
    )


def _render_top_profile(
    profile: tuple[tuple[float, float], ...],
    shape: tuple[int, int],
    center_x_px: float,
    start_y_px: int,
    units_per_pixel: float,
    base_z: float,
) -> np.ndarray:
    polygon = np.rint(
        np.asarray(
            [
                (
                    center_x_px + x / units_per_pixel,
                    start_y_px + (z - base_z) / units_per_pixel,
                )
                for x, z in profile
            ]
        )
    ).astype(np.int32)
    rendered = np.zeros(shape, np.uint8)
    cv2.fillPoly(rendered, [polygon], 255)
    return rendered


def _overlay(image: np.ndarray, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    result = image.copy()
    kernel = np.ones((3, 3), np.uint8)
    target_edge = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel) > 0
    rendered_edge = cv2.morphologyEx(rendered, cv2.MORPH_GRADIENT, kernel) > 0
    result[target_edge] = (255, 45, 141)
    result[rendered_edge] = (255, 247, 214)
    return result


def _report_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _render_audit(root: Path, objects: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 4, figsize=(16, 18), facecolor="#f4f4f4")
    for axis, item in zip(axes.flat, objects):
        front = cv2.cvtColor(cv2.imread(str(root / item["source_images"]["front"])), cv2.COLOR_BGR2RGB)
        top = cv2.cvtColor(cv2.imread(str(root / item["source_images"]["top"])), cv2.COLOR_BGR2RGB)
        if item["evaluation_status"] == "held_out_not_scored":
            composite = np.hstack([front, top])
            axis.imshow(composite, alpha=0.40)
            axis.text(0.5, 0.5, "HELD OUT · NOT SCORED", transform=axis.transAxes,
                      ha="center", va="center", fontsize=10, fontweight="bold")
            detail = "frozen test split"
        else:
            front_target = cv2.imread(str(root / item["artifacts"]["front_target"]), cv2.IMREAD_GRAYSCALE)
            front_render = cv2.imread(str(root / item["artifacts"]["front_render"]), cv2.IMREAD_GRAYSCALE)
            top_target = cv2.imread(str(root / item["artifacts"]["top_target"]), cv2.IMREAD_GRAYSCALE)
            top_render = cv2.imread(str(root / item["artifacts"]["top_render"]), cv2.IMREAD_GRAYSCALE)
            composite = np.hstack([
                _overlay(front, front_target, front_render),
                _overlay(top, top_target, top_render),
            ])
            axis.imshow(composite)
            detail = f"front {item['metrics']['front_profile_iou']:.3f} · top env {item['metrics']['top_envelope_iou']:.3f}"
        axis.set_title(f"{item['object_id'].upper()} · {item['split'].upper()}\n{detail}",
                       loc="left", fontsize=8.5, fontweight="bold")
        axis.set_axis_off()
    figure.suptitle("PHASE 5.2 · EXACT RING HEAD-SUPPORT ENVELOPES",
                    fontsize=17, fontweight="bold", color="#20252b")
    figure.text(
        0.5, 0.012,
        "Front + top · pink = machine target · white = exact support envelope · relief/openwork excluded",
        ha="center", fontsize=10.5, color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _render_geometry_preview(shape: Any, output_path: Path, title: str) -> None:
    """Render the actual tessellated B-rep from four diagnostic cameras."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, triangles = shape.tessellate(0.12)
    points = np.asarray(
        [(float(vertex.x), float(vertex.y), float(vertex.z)) for vertex in vertices],
        dtype=np.float64,
    )
    faces = np.asarray(triangles, dtype=np.int64)
    mesh = points[faces]
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float((high - low).max()) / 2.0, 1.0)
    cameras = (
        ("ISOMETRIC", 24, -48),
        ("ORNAMENT FACE", 90, -90),
        ("RING PROFILE", 0, -90),
        ("OPPOSITE", 24, 132),
    )
    figure = plt.figure(figsize=(12, 10), facecolor="#f4f4f4")
    for index, (label, elevation, azimuth) in enumerate(cameras, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        collection = Poly3DCollection(
            mesh,
            facecolor="#b7d3dc",
            edgecolor="#6d8790",
            linewidth=0.04,
            alpha=1.0,
            rasterized=True,
        )
        axis.add_collection3d(collection)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(label, loc="left", fontsize=10, fontweight="bold")
        axis.set_axis_off()
    figure.suptitle(title, fontsize=15, fontweight="bold", color="#20252b")
    figure.text(
        0.5,
        0.02,
        "Actual tessellated STEP candidate · structural envelope only · relief and openings excluded",
        ha="center",
        fontsize=10,
        color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _mean(objects: list[dict[str, Any]], key: str) -> float | None:
    values = [item["metrics"][key] for item in objects if item["metrics"] is not None]
    return round(float(np.mean(values)), 6) if values else None


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    import cadquery as cq

    parameter_report = json.loads(
        (root / "data" / "phase5_stl1_ring_parameters_v1" / "report.json").read_text(encoding="utf-8")
    )
    body_report = json.loads(
        (root / "data" / "phase5_stl1_exact_ring_body_fit_v1" / "report.json").read_text(encoding="utf-8")
    )
    body_by_id = {item["object_id"]: item for item in body_report["objects"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    for source in parameter_report["objects"]:
        object_id = source["object_id"]
        ring_spec = RingBodySpec(**body_by_id[object_id]["ring_body_spec"])
        inner_profile, outer_profile = _profile_controls(
            source["ring_parameter_evidence"],
            source["normalized_parameter_proposal"]["shank"],
        )
        ring_body = build_profiled_ring_body(cq, ring_spec, inner_profile, outer_profile)
        dimensions = _head_dimensions(source, root, ring_spec)
        front_mask = cv2.imread(
            str(root / source["views"]["front"]["mask_path"]), cv2.IMREAD_GRAYSCALE
        ) > 127
        front_profile = _front_profile(
            front_mask, dimensions["head_width"], dimensions["head_depth"]
        )
        support_spec = HeadSupportSpec(
            component_id=f"{object_id}_head_support_001",
            height=dimensions["head_height"],
            base_z=ring_spec.inner_height / 2.0 + ring_spec.radial_thickness * 0.15,
            shoulder_radius=max(0.35, ring_spec.radial_thickness * 0.62),
        )
        top_mask = cv2.imread(
            str(root / source["views"]["top"]["mask_path"]), cv2.IMREAD_GRAYSCALE
        ) > 127
        top_target = np.zeros_like(top_mask, dtype=np.uint8)
        start_y = int(dimensions["top_start_y_px"])
        top_target[start_y:] = top_mask[start_y:].astype(np.uint8) * 255
        diameter_px = float(
            source["ring_parameter_evidence"]["aperture"]["equivalent_diameter_px"]
        )
        units_per_pixel = NOMINAL_INNER_DIAMETER / diameter_px
        top_profile = _top_profile(
            top_target > 0,
            dimensions["top_center_x_px"],
            start_y,
            units_per_pixel,
            support_spec.base_z,
        )
        support = build_head_support_envelope(
            cq, ring_body, support_spec, front_profile, top_profile
        )

        front_bbox = _bbox(front_mask)
        front_render = _render_front_profile(
            front_profile,
            front_mask.shape,
            front_bbox,
            dimensions["head_width"],
            dimensions["head_depth"],
        )
        front_target = front_mask.astype(np.uint8) * 255
        top_render = _render_top_profile(
            top_profile,
            top_mask.shape,
            dimensions["top_center_x_px"],
            start_y,
            units_per_pixel,
            support_spec.base_z,
        )

        object_dir = output_dir / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "front_target": object_dir / "front_head_target.png",
            "front_render": object_dir / "front_head_support_render.png",
            "top_target": object_dir / "top_head_target.png",
            "top_render": object_dir / "top_head_support_render.png",
            "step": object_dir / f"{object_id}_shank_head_support.step",
        }
        for key in ("front_target", "front_render", "top_target", "top_render"):
            cv2.imwrite(str(paths[key]), locals()[key])
        cq.exporters.export(cq.Workplane(obj=support["metal"]), str(paths["step"]))
        if source["split"] == "val":
            paths["geometry_preview"] = object_dir / f"{object_id}_geometry_preview.png"
            _render_geometry_preview(
                support["metal"],
                paths["geometry_preview"],
                f"PHASE 5.2 · {object_id.upper()} HEAD-SUPPORT CANDIDATE",
            )
        scored = source["split"] != "test"
        metrics = None
        if scored:
            metrics = {
                "front_profile_iou": round(_iou(front_target, front_render), 6),
                "front_boundary_f1": round(_boundary_f1(front_target, front_render), 6),
                "top_envelope_iou": round(_iou(top_target, top_render), 6),
                "top_envelope_boundary_f1": round(_boundary_f1(top_target, top_render), 6),
            }
        objects.append({
            "object_id": object_id,
            "category": "ring",
            "split": source["split"],
            "evaluation_status": "scored" if scored else "held_out_not_scored",
            "source_images": {
                "front": source["views"]["front"]["image_path"],
                "top": source["views"]["top"]["image_path"],
            },
            "ring_body_spec": asdict(ring_spec),
            "head_support_spec": asdict(support_spec),
            "head_dimensions_nominal": dimensions,
            "profile_control_count": {
                "front": len(front_profile),
                "top": len(top_profile),
            },
            "exact_geometry": {
                "head_support_valid": bool(support["head_support"].isValid()),
                "shoulder_count": len(support["shoulders"]),
                "shoulders_valid": all(shape.isValid() for shape in support["shoulders"]),
                "metal_valid": bool(support["metal"].isValid()),
                "metal_solid_count": len(support["metal"].Solids()),
            },
            "metrics": metrics,
            "artifacts": {key: _report_path(path, root) for key, path in paths.items()},
        })

    audit_path = output_dir / "stl1_exact_head_support_fit_audit.png"
    _render_audit(root, objects, audit_path)
    train = [item for item in objects if item["split"] == "train"]
    validation = [item for item in objects if item["split"] == "val"]
    test = [item for item in objects if item["split"] == "test"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_profile": "ring_only_multiview_exact_head_support_envelope",
        "scope": {
            "included": [
                "exact shank",
                "front/top two-view head-support visual hull",
                "two shoulder supports",
            ],
            "excluded": ["surface relief", "openwork", "stones", "prongs", "seats", "cavities", "manufacturing scale"],
            "evidence_views": ["front", "top", "lsv", "rsv"],
        },
        "summary": {
            "object_count": len(objects),
            "train_scored_count": len(train),
            "validation_scored_count": len(validation),
            "test_held_out_count": len(test),
            "train_mean_front_profile_iou": _mean(train, "front_profile_iou"),
            "validation_mean_front_profile_iou": _mean(validation, "front_profile_iou"),
            "train_mean_top_envelope_iou": _mean(train, "top_envelope_iou"),
            "validation_mean_top_envelope_iou": _mean(validation, "top_envelope_iou"),
        },
        "checks": {
            "all_24_support_candidates_generated": len(objects) == 24,
            "all_head_supports_valid": all(item["exact_geometry"]["head_support_valid"] for item in objects),
            "all_48_shoulders_valid": all(item["exact_geometry"]["shoulder_count"] == 2 and item["exact_geometry"]["shoulders_valid"] for item in objects),
            "all_metals_valid_single_solids": all(item["exact_geometry"]["metal_valid"] and item["exact_geometry"]["metal_solid_count"] == 1 for item in objects),
            "all_step_exports_present": all((root / item["artifacts"]["step"]).is_file() for item in objects),
            "all_validation_geometry_previews_present": all(
                (root / item["artifacts"]["geometry_preview"]).is_file()
                for item in validation
            ),
            "held_out_test_not_scored": len(test) == 3 and all(item["metrics"] is None for item in test),
            "ring_only_scope_enforced": all(item["category"] == "ring" for item in objects),
            "detail_and_metric_accuracy_not_claimed": True,
        },
        "objects": objects,
        "artifacts": {"audit": _report_path(audit_path, root)},
        "decision": "two_view_head_support_envelope_ready_for_relief_and_openwork_reconstruction",
        "claims": {
            "image_derived_exact_head_support_generated": True,
            "complete_ring_head_reconstructed": False,
            "multi_view_calibrated_geometry_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The front profile and dimensions are machine-derived from the same silhouettes used for fitting scores.",
            "Top and side views provide envelope ratios without calibrated cross-view scale or camera intrinsics.",
            "Left/right side estimates are retained as disagreement diagnostics and do not override the direct front-profile aspect ratio.",
            "The head blank intentionally fills openwork and omits all sculptural relief.",
            "The result is a structural support candidate, not a visually complete ring.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase5_stl1_exact_head_support_fit_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(root, output_dir)
    print(json.dumps({
        "decision": report["decision"],
        "summary": report["summary"],
        "checks": report["checks"],
        "artifacts": report["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
