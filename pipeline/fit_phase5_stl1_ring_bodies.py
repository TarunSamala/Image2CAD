"""Fit exact elliptical shank bodies to STL-1 top-view ring evidence.

This is the first image-derived exact-CAD fitting checkpoint.  It reconstructs
only the shank aperture/profile; shoulders, head, gallery, relief, stones and
settings remain excluded.  The fixed test split is generated but not scored.
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

from pipeline.exact_cad_v2 import RingBodySpec, build_profiled_ring_body


SCHEMA_VERSION = "lalitha_phase5_stl1_exact_ring_body_fit_v1"
NOMINAL_INNER_DIAMETER = 16.6


def _shape_to_mesh(shape: Any):
    import trimesh

    vertices, triangles = shape.tessellate(0.06)
    points = np.asarray(
        [(float(vertex.x), float(vertex.y), float(vertex.z)) for vertex in vertices],
        dtype=np.float64,
    )
    faces = np.asarray(triangles, dtype=np.int64)
    return trimesh.Trimesh(vertices=points, faces=faces, process=True)


def _render_top_profile(
    body: dict[str, Any],
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    pixels_per_unit: float,
) -> np.ndarray:
    rendered = np.zeros(shape, np.uint8)
    polygons: dict[str, np.ndarray] = {}
    for name in ("outer_wire", "inner_wire"):
        edge = body[name].Edges()[0]
        points = [
            edge.positionAt(float(parameter))
            for parameter in np.linspace(0.0, 1.0, 256, endpoint=False)
        ]
        polygons[name] = np.rint(np.asarray([
            (
                center_xy[0] + float(point.x) * pixels_per_unit,
                center_xy[1] + float(point.z) * pixels_per_unit,
            )
            for point in points
        ])).astype(np.int32)
    cv2.fillPoly(rendered, [polygons["outer_wire"]], 255)
    cv2.fillPoly(rendered, [polygons["inner_wire"]], 0)
    return rendered


def _profile_controls(
    evidence: dict[str, Any],
    shank: dict[str, Any],
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    aperture = evidence["aperture"]
    cx, cy = aperture["center_xy_px"]
    diameter_px = float(aperture["equivalent_diameter_px"])
    units_per_pixel = NOMINAL_INNER_DIAMETER / diameter_px
    cap_px = shank["retained_max_thickness_units"] * diameter_px
    inner_controls: list[tuple[float, float]] = []
    outer_controls: list[tuple[float, float]] = []
    for sample in evidence["radial_samples"][::3]:
        angle = np.deg2rad(float(sample["angle_deg"]))
        inner_x = float(sample["inner_x_px"])
        inner_y = float(sample["inner_y_px"])
        thickness = min(float(sample["thickness_px"]), cap_px)
        outer_x = inner_x + thickness * float(np.cos(angle))
        outer_y = inner_y + thickness * float(np.sin(angle))
        inner_controls.append(
            ((inner_x - cx) * units_per_pixel, (inner_y - cy) * units_per_pixel)
        )
        outer_controls.append(
            ((outer_x - cx) * units_per_pixel, (outer_y - cy) * units_per_pixel)
        )
    return tuple(inner_controls), tuple(outer_controls)


def _aperture_mask(mask: np.ndarray) -> np.ndarray:
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    regions: list[tuple[int, int]] = []
    height, width = mask.shape
    for label in range(1, count):
        x, y, box_width, box_height, area = [int(value) for value in stats[label]]
        if x == 0 or y == 0 or x + box_width >= width or y + box_height >= height:
            continue
        regions.append((area, label))
    if not regions:
        raise ValueError("No enclosed shank aperture")
    label = max(regions)[1]
    return labels == label


def _shank_target(
    jewelry_mask: np.ndarray,
    aperture: np.ndarray,
    radial_limit_px: float,
) -> np.ndarray:
    outside_aperture = (~aperture).astype(np.uint8)
    distance = cv2.distanceTransform(outside_aperture, cv2.DIST_L2, 5)
    target = jewelry_mask & (distance <= radial_limit_px) & (~aperture)
    return target.astype(np.uint8) * 255


def _iou(target: np.ndarray, rendered: np.ndarray) -> float:
    target_fg = target > 0
    rendered_fg = rendered > 0
    intersection = np.count_nonzero(target_fg & rendered_fg)
    union = np.count_nonzero(target_fg | rendered_fg)
    return float(intersection / max(union, 1))


def _boundary_f1(target: np.ndarray, rendered: np.ndarray, tolerance_px: int = 3) -> float:
    kernel = np.ones((3, 3), np.uint8)
    target_edge = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel) > 0
    rendered_edge = cv2.morphologyEx(rendered, cv2.MORPH_GRADIENT, kernel) > 0
    if not target_edge.any() or not rendered_edge.any():
        return 0.0
    target_near = cv2.dilate(target_edge.astype(np.uint8), np.ones((2 * tolerance_px + 1,) * 2, np.uint8)) > 0
    rendered_near = cv2.dilate(rendered_edge.astype(np.uint8), np.ones((2 * tolerance_px + 1,) * 2, np.uint8)) > 0
    precision = np.count_nonzero(rendered_edge & target_near) / np.count_nonzero(rendered_edge)
    recall = np.count_nonzero(target_edge & rendered_near) / np.count_nonzero(target_edge)
    return float(2.0 * precision * recall / max(precision + recall, 1e-9))


def _overlay(image: np.ndarray, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    result = image.copy()
    target_edge = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    rendered_edge = cv2.morphologyEx(rendered, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    result[target_edge] = (255, 45, 141)
    result[rendered_edge] = (255, 247, 214)
    return result


def _export_step(cq: Any, body: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane(obj=body["shank"]), str(path))


def _render_audit(root: Path, objects: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 4, figsize=(14, 19), facecolor="#f4f4f4")
    for axis, item in zip(axes.flat, objects):
        image = cv2.cvtColor(
            cv2.imread(str(root / item["source_top_image"])), cv2.COLOR_BGR2RGB
        )
        if item["evaluation_status"] == "held_out_not_scored":
            axis.imshow(image, alpha=0.45)
            axis.text(
                0.5, 0.5, "HELD OUT\nNOT SCORED", transform=axis.transAxes,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="#20252b",
            )
            score_text = "frozen test split"
        else:
            target = cv2.imread(str(root / item["artifacts"]["target_mask"]), cv2.IMREAD_GRAYSCALE)
            rendered = cv2.imread(str(root / item["artifacts"]["rendered_mask"]), cv2.IMREAD_GRAYSCALE)
            axis.imshow(_overlay(image, target, rendered))
            score_text = f"IoU {item['metrics']['shank_iou']:.3f} · F1 {item['metrics']['boundary_f1']:.3f}"
        axis.set_title(
            f"{item['object_id'].upper()} · {item['split'].upper()}\n{score_text}",
            loc="left", fontsize=9, fontweight="bold",
        )
        axis.set_axis_off()
    figure.suptitle(
        "PHASE 5.1 · STL-1 IMAGE-DERIVED EXACT RING-BODY FIT",
        fontsize=17, fontweight="bold", color="#20252b",
    )
    figure.text(
        0.5, 0.012,
        "Pink = measured shank target · white = exact B-rep render · top view only · no head/detail claim",
        ha="center", fontsize=10.5, color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item["metrics"][key] for item in items if item["metrics"] is not None]
    return round(float(np.mean(values)), 6) if values else None


def _report_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    import cadquery as cq

    parameter_report = json.loads(
        (root / "data" / "phase5_stl1_ring_parameters_v1" / "report.json")
        .read_text(encoding="utf-8")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    for source in parameter_report["objects"]:
        proposal = source["normalized_parameter_proposal"]
        aperture = source["ring_parameter_evidence"]["aperture"]
        shank = proposal["shank"]
        inner_width = NOMINAL_INNER_DIAMETER * proposal["inner_aperture"]["major_axis_units"]
        inner_height = NOMINAL_INNER_DIAMETER * proposal["inner_aperture"]["minor_axis_units"]
        radial_thickness = NOMINAL_INNER_DIAMETER * shank["median_thickness_units"]
        spec = RingBodySpec(
            object_id=source["object_id"],
            inner_width=inner_width,
            inner_height=inner_height,
            radial_thickness=radial_thickness,
            axial_depth=max(0.75, radial_thickness * 0.92),
        )
        inner_profile, outer_profile = _profile_controls(
            source["ring_parameter_evidence"], shank
        )
        body = build_profiled_ring_body(cq, spec, inner_profile, outer_profile)
        top_view = source["views"]["top"]
        jewelry_mask = cv2.imread(
            str(root / top_view["mask_path"]), cv2.IMREAD_GRAYSCALE
        )
        if jewelry_mask is None:
            raise FileNotFoundError(top_view["mask_path"])
        aperture_mask = _aperture_mask(jewelry_mask)
        aperture_diameter_px = float(aperture["equivalent_diameter_px"])
        target = _shank_target(
            jewelry_mask > 127,
            aperture_mask,
            shank["retained_max_thickness_units"] * aperture_diameter_px,
        )
        rendered = _render_top_profile(
            body,
            jewelry_mask.shape,
            tuple(aperture["center_xy_px"]),
            aperture_diameter_px / NOMINAL_INNER_DIAMETER,
        )
        object_dir = output_dir / "objects" / source["object_id"]
        object_dir.mkdir(parents=True, exist_ok=True)
        target_path = object_dir / "top_shank_target.png"
        rendered_path = object_dir / "top_exact_ring_body_render.png"
        step_path = object_dir / f"{source['object_id']}_ring_body.step"
        cv2.imwrite(str(target_path), target)
        cv2.imwrite(str(rendered_path), rendered)
        _export_step(cq, body, step_path)
        scored = source["split"] != "test"
        metrics = None
        if scored:
            metrics = {
                "shank_iou": round(_iou(target, rendered), 6),
                "boundary_f1": round(_boundary_f1(target, rendered), 6),
            }
        objects.append({
            "object_id": source["object_id"],
            "category": "ring",
            "split": source["split"],
            "evaluation_status": "scored" if scored else "held_out_not_scored",
            "source_top_image": top_view["image_path"],
            "ring_body_spec": asdict(spec),
            "profile_control_count": body["profile_control_count"],
            "exact_geometry": {
                "valid": bool(body["shank"].isValid()),
                "solid_count": len(body["shank"].Solids()),
                "volume_nominal_units3": round(float(body["shank"].Volume()), 6),
            },
            "metrics": metrics,
            "artifacts": {
                "target_mask": _report_path(target_path, root),
                "rendered_mask": _report_path(rendered_path, root),
                "step": _report_path(step_path, root),
            },
        })

    audit_path = output_dir / "stl1_exact_ring_body_fit_audit.png"
    _render_audit(root, objects, audit_path)
    train = [item for item in objects if item["split"] == "train"]
    validation = [item for item in objects if item["split"] == "val"]
    test = [item for item in objects if item["split"] == "test"]
    all_exact_valid = all(
        item["exact_geometry"]["valid"]
        and item["exact_geometry"]["solid_count"] == 1
        for item in objects
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_profile": "ring_only_image_derived_exact_shank_fit",
        "scope": {
            "included": ["ring inner aperture", "elliptical shank envelope"],
            "excluded": ["shoulders", "head", "gallery", "relief", "stones", "prongs", "seats", "cavities"],
            "scored_view": "top",
            "scale": "nominal display scale; not verified millimetres",
        },
        "summary": {
            "object_count": len(objects),
            "train_scored_count": len(train),
            "validation_scored_count": len(validation),
            "test_held_out_count": len(test),
            "train_mean_shank_iou": _mean(train, "shank_iou"),
            "validation_mean_shank_iou": _mean(validation, "shank_iou"),
            "train_mean_boundary_f1": _mean(train, "boundary_f1"),
            "validation_mean_boundary_f1": _mean(validation, "boundary_f1"),
        },
        "checks": {
            "all_24_ring_bodies_generated": len(objects) == 24,
            "all_exact_breps_valid_single_solids": all_exact_valid,
            "all_step_exports_present": all(
                (root / item["artifacts"]["step"]).is_file() for item in objects
            ),
            "train_and_validation_scored": len(train) == 18 and len(validation) == 3,
            "held_out_test_not_scored": len(test) == 3 and all(item["metrics"] is None for item in test),
            "ring_only_scope_enforced": all(item["category"] == "ring" for item in objects),
            "metric_accuracy_not_claimed": True,
        },
        "objects": objects,
        "artifacts": {"audit": _report_path(audit_path, root)},
        "decision": "exact_ring_body_baseline_ready_for_shoulder_gallery_fitting",
        "claims": {
            "image_derived_exact_shank_generated": True,
            "five_view_ring_fit_validated": False,
            "complete_ring_cad_validated": False,
            "metric_accuracy_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Only the top-view shank region is fitted and scored in this checkpoint.",
            "The retained STEP files use a nominal display scale because STL-1 provides no physical dimension.",
            "The held-out test split is generated for pipeline integrity but remains unscored until the fitting policy is frozen.",
            "The shank target and spline initialization come from the same machine silhouette, so the score measures fitting consistency rather than independent geometric truth.",
            "A complete ring requires multi-view shoulders, gallery, head, relief and component reconstruction.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/phase5_stl1_exact_ring_body_fit_v1"),
    )
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
