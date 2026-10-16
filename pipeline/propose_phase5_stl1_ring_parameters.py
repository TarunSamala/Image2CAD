"""Propose ring-only, scale-normalized CAD parameters from STL-1 evidence.

This adapter is deliberately conservative.  It measures retained silhouettes
from every view and estimates the shank aperture/profile from the STL-1 ``top``
view, but it never converts image measurements to millimetres and never marks a
proposal as CAD-authorized.  The output is the dataset bridge between image
evidence and later render-and-compare fitting of exact ring generators.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "lalitha_phase5_stl1_ring_parameter_proposals_v1"
VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Foreground mask is empty")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def _enclosed_regions(mask: np.ndarray) -> list[dict[str, Any]]:
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    height, width = mask.shape
    regions: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, region_width, region_height, area = [
            int(value) for value in stats[label]
        ]
        touches_border = (
            x == 0 or y == 0 or x + region_width >= width or y + region_height >= height
        )
        if touches_border or area < 16:
            continue
        regions.append({
            "label": label,
            "area_px": area,
            "bbox_xywh": [x, y, region_width, region_height],
            "centroid_xy": [float(value) for value in centroids[label]],
            "mask": labels == label,
        })
    return sorted(regions, key=lambda item: item["area_px"], reverse=True)


def _fit_aperture(region: dict[str, Any]) -> dict[str, Any]:
    binary = region["mask"].astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise ValueError("Largest enclosed region cannot support an ellipse fit")
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    major = float(max(axis_a, axis_b))
    minor = float(min(axis_a, axis_b))
    equivalent_diameter = 2.0 * math.sqrt(float(region["area_px"]) / math.pi)
    return {
        "center_xy_px": [round(float(cx), 4), round(float(cy), 4)],
        "major_axis_px": round(major, 4),
        "minor_axis_px": round(minor, 4),
        "equivalent_diameter_px": round(equivalent_diameter, 4),
        "aspect_ratio": round(major / max(minor, 1e-6), 6),
        "ellipse_angle_deg": round(float(angle), 4),
        "area_px": int(region["area_px"]),
        "contour": contour,
    }


def _radial_shank_samples(
    mask: np.ndarray, center_xy: tuple[float, float], aperture_diameter: float
) -> list[dict[str, float]]:
    """Measure the first foreground band reached from the aperture centre."""
    height, width = mask.shape
    cx, cy = center_xy
    max_radius = int(math.ceil(math.hypot(width, height)))
    samples: list[dict[str, float]] = []
    for angle_deg in np.linspace(0.0, 360.0, 96, endpoint=False):
        angle = math.radians(float(angle_deg))
        values: list[bool] = []
        coordinates: list[tuple[int, int]] = []
        for radius in range(max_radius):
            x = int(round(cx + radius * math.cos(angle)))
            y = int(round(cy + radius * math.sin(angle)))
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            values.append(bool(mask[y, x]))
            coordinates.append((x, y))
        starts = [index for index in range(1, len(values)) if values[index] and not values[index - 1]]
        if not starts:
            continue
        start = starts[0]
        end = start
        while end + 1 < len(values) and values[end + 1]:
            end += 1
        thickness = float(end - start + 1)
        if thickness < 1.0:
            continue
        samples.append({
            "angle_deg": round(float(angle_deg), 3),
            "inner_radius_px": float(start),
            "thickness_px": thickness,
            "thickness_units": thickness / max(aperture_diameter, 1e-6),
            "inner_x_px": float(coordinates[start][0]),
            "inner_y_px": float(coordinates[start][1]),
            "outer_x_px": float(coordinates[end][0]),
            "outer_y_px": float(coordinates[end][1]),
        })
    return samples


def _robust_shank_summary(samples: list[dict[str, float]]) -> dict[str, Any]:
    if len(samples) < 16:
        raise ValueError("Insufficient radial shank observations")
    values = np.asarray([item["thickness_units"] for item in samples], dtype=np.float64)
    # The ornament/head occupies a contiguous arc and can be several times
    # thicker than the shank. The lower half of radial bands is a conservative
    # estimator of the repeated shank profile without assuming which image side
    # contains the head.
    cutoff = float(np.quantile(values, 0.50))
    retained = values[values <= cutoff]
    return {
        "sample_count": int(len(samples)),
        "robust_sample_count": int(len(retained)),
        "median_thickness_units": round(float(np.median(retained)), 6),
        "p10_thickness_units": round(float(np.quantile(retained, 0.10)), 6),
        "p90_thickness_units": round(float(np.quantile(retained, 0.90)), 6),
        "dispersion_units": round(
            float(np.quantile(retained, 0.90) - np.quantile(retained, 0.10)), 6
        ),
        "retained_max_thickness_units": round(cutoff, 6),
        "outlier_rejection": "upper_50_percent_removed_to_reduce_head_and_ornament_bias",
    }


def _view_measurement(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    x, y, box_width, box_height = _bbox(mask)
    regions = _enclosed_regions(mask)
    return {
        "image_size_wh": [width, height],
        "foreground_bbox_xywh": [x, y, box_width, box_height],
        "foreground_fraction": round(float(mask.mean()), 8),
        "bbox_width_fraction": round(box_width / width, 8),
        "bbox_height_fraction": round(box_height / height, 8),
        "bbox_aspect_ratio": round(box_width / max(box_height, 1), 8),
        "enclosed_region_count": len(regions),
        "largest_enclosed_area_fraction": round(
            (regions[0]["area_px"] / float(width * height)) if regions else 0.0, 8
        ),
    }


def _parameter_proposal(top_mask: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    regions = _enclosed_regions(top_mask)
    if not regions:
        raise ValueError("Top view has no enclosed ring aperture")
    aperture = _fit_aperture(regions[0])
    diameter = float(aperture["equivalent_diameter_px"])
    samples = _radial_shank_samples(
        top_mask, tuple(aperture["center_xy_px"]), diameter
    )
    shank = _robust_shank_summary(samples)
    proposal = {
        "parameter_space": "scale_normalized_ring_units",
        "scale_reference": "top_view_inner_aperture_equivalent_diameter_equals_1",
        "inner_aperture": {
            "major_axis_units": round(float(aperture["major_axis_px"]) / diameter, 6),
            "minor_axis_units": round(float(aperture["minor_axis_px"]) / diameter, 6),
            "aspect_ratio": aperture["aspect_ratio"],
            "ellipse_angle_deg": aperture["ellipse_angle_deg"],
        },
        "shank": shank,
        "absolute_dimensions": None,
        "unit": None,
        "cad_authorized": False,
        "manufacturing_authorized": False,
    }
    evidence = {
        "aperture": {key: value for key, value in aperture.items() if key != "contour"},
        "radial_samples": samples,
    }
    return proposal, evidence


def _render_audit(root: Path, objects: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    figure, axes = plt.subplots(6, 4, figsize=(14, 19), facecolor="#f4f4f4")
    for axis, item in zip(axes.flat, objects):
        image = cv2.cvtColor(
            cv2.imread(str(root / item["views"]["top"]["image_path"])),
            cv2.COLOR_BGR2RGB,
        )
        evidence = item["ring_parameter_evidence"]
        aperture = evidence["aperture"]
        axis.imshow(image)
        ellipse = Ellipse(
            aperture["center_xy_px"],
            aperture["major_axis_px"],
            aperture["minor_axis_px"],
            angle=aperture["ellipse_angle_deg"],
            fill=False,
            edgecolor="#ff2d8d",
            linewidth=1.8,
        )
        axis.add_patch(ellipse)
        proposal = item["normalized_parameter_proposal"]
        retained_max = proposal["shank"]["retained_max_thickness_units"]
        retained_samples = [
            sample for sample in evidence["radial_samples"]
            if sample["thickness_units"] <= retained_max
        ]
        for sample in retained_samples[::4]:
            axis.plot(
                [sample["inner_x_px"], sample["outer_x_px"]],
                [sample["inner_y_px"], sample["outer_y_px"]],
                color="#fff7d6",
                linewidth=1.0,
            )
        axis.set_title(
            f"{item['object_id'].upper()} · {item['split'].upper()}\n"
            f"shank {proposal['shank']['median_thickness_units']:.3f} ring units",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        axis.set_axis_off()
    figure.suptitle(
        "STL-1 · RING-ONLY IMAGE EVIDENCE → NORMALIZED CAD PARAMETERS",
        fontsize=17,
        fontweight="bold",
        color="#20252b",
    )
    figure.text(
        0.5,
        0.012,
        "24 rings · pink = detected inner aperture · white = robust shank samples · no millimetres inferred",
        ha="center",
        fontsize=10.5,
        color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _report_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    object_paths = sorted((root / "dataset" / "prepared_v1" / "objects").glob("ring_*.json"))
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    for object_path in object_paths:
        source = json.loads(object_path.read_text(encoding="utf-8"))
        if source.get("category") != "ring":
            raise ValueError(f"Non-ring object rejected: {source.get('object_id')}")
        views: dict[str, Any] = {}
        for view in VIEW_ORDER:
            source_view = source["views"][view]
            mask = cv2.imread(
                str(root / source_view["jewelry_mask_path"]), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise FileNotFoundError(source_view["jewelry_mask_path"])
            views[view] = {
                "image_path": source_view["image_path"],
                "mask_path": source_view["jewelry_mask_path"],
                "measurement": _view_measurement(mask > 127),
            }
        top_mask = cv2.imread(
            str(root / source["views"]["top"]["jewelry_mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        ) > 127
        proposal, evidence = _parameter_proposal(top_mask)
        objects.append({
            "object_id": source["object_id"],
            "source_object_name": source["source_object_name"],
            "category": "ring",
            "split": source["split"],
            "views": views,
            "normalized_parameter_proposal": proposal,
            "ring_parameter_evidence": evidence,
        })

    audit_path = output_dir / "stl1_ring_parameter_audit.png"
    _render_audit(root, objects, audit_path)
    split_counts = {
        split: sum(item["split"] == split for item in objects)
        for split in ("train", "val", "test")
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_profile": "ring_only_image_evidence_parameter_proposals",
        "source": {
            "dataset": "STL-1",
            "prepared_objects": "dataset/prepared_v1/objects",
        },
        "summary": {
            "object_count": len(objects),
            "view_count": len(objects) * len(VIEW_ORDER),
            "category_count": 1,
            "categories": ["ring"],
            "split_counts": split_counts,
            "parameterized_ring_count": len(objects),
        },
        "checks": {
            "all_24_stl1_rings_processed": len(objects) == 24,
            "all_120_views_measured": len(objects) * len(VIEW_ORDER) == 120,
            "ring_only_scope_enforced": all(item["category"] == "ring" for item in objects),
            "all_top_apertures_detected": len(objects) == 24,
            "no_metric_units_fabricated": all(
                item["normalized_parameter_proposal"]["unit"] is None
                and item["normalized_parameter_proposal"]["absolute_dimensions"] is None
                for item in objects
            ),
            "no_image_proposal_cad_authorized": not any(
                item["normalized_parameter_proposal"]["cad_authorized"]
                for item in objects
            ),
            "split_identity_preserved": split_counts == {"train": 18, "val": 3, "test": 3},
        },
        "objects": objects,
        "artifacts": {
            "audit": _report_path(audit_path, root),
        },
        "decision": "dataset_ring_parameter_bridge_ready_for_render_and_compare_fitting",
        "claims": {
            "dataset_used": True,
            "ring_only_scope": True,
            "scale_normalized_parameters_proposed": True,
            "image_derived_exact_cad_validated": False,
            "metric_accuracy_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "STL-1 supplies images and machine silhouettes but no physical scale or production CAD target.",
            "The largest enclosed top-view region is treated as the shank aperture and remains a machine proposal.",
            "Sculptural relief, stones, prongs and hidden surfaces require component tracks and render-and-compare fitting.",
            "Perfect or manufacturing-verified geometry cannot be asserted from this dataset alone.",
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
        "--output-dir",
        type=Path,
        default=Path("data/phase5_stl1_ring_parameters_v1"),
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
