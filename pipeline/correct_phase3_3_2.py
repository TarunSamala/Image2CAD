"""Phase 3.3.2: remove false missing-prong previews without distorting CAD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from reconstruct_phase3 import CAMERAS
from refine_phase3_1 import _jewelry_transform
from refine_phase3_3 import _camera_with_azimuth
from validate_phase3_2 import _basis_camera, _clean_render


PRONG_IDS = ("prong_ne", "prong_nw", "prong_sw", "prong_se")


def _project_tip_positions(prongs: list[dict], camera) -> dict[str, tuple[float, float]]:
    points = {
        record["prong_id"]: np.asarray(record["tip"], dtype=float)
        for record in prongs
    }
    common_z = float(np.mean([point[2] for point in points.values()]))
    center = np.array((0.0, 0.0, common_z))
    right, up = np.asarray(camera.right), np.asarray(camera.up)
    return {
        prong_id: (float((point - center) @ right), float((point - center) @ up))
        for prong_id, point in points.items()
    }


def _visibility_metrics(prongs: list[dict], camera, cap_radius: float) -> dict:
    projected = _project_tip_positions(prongs, camera)
    ids = list(projected)
    distances = []
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            distances.append(float(np.linalg.norm(np.subtract(projected[first], projected[second]))))
    minimum_separation = min(distances)
    minimum_axis_clearance = min(abs(point[0]) for point in projected.values())
    return {
        "projected_tip_positions": {
            prong_id: [round(value, 6) for value in point]
            for prong_id, point in projected.items()
        },
        "minimum_pair_separation": round(minimum_separation, 6),
        "minimum_pair_separation_cap_diameters": round(minimum_separation / (2 * cap_radius), 6),
        "minimum_center_axis_clearance_cap_radii": round(minimum_axis_clearance / cap_radius, 6),
        "four_distinct_tip_positions": len(set(projected.values())) == 4,
        "no_tip_on_center_axis": minimum_axis_clearance >= cap_radius * 0.45,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correct(source_dir: Path, output_dir: Path) -> dict:
    build = json.loads((source_dir / "phase3_3_1_build.json").read_text(encoding="utf-8"))
    validation = json.loads((source_dir / "phase3_3_1_validation.json").read_text(encoding="utf-8"))
    model_path = source_dir / "ring01_phase3_3_1.stl"
    step_path = source_dir / "ring01_phase3_3_1.step"
    mesh = trimesh.load(model_path, force="mesh")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The former +/-45-degree inspection directions were collinear with two
    # diagonal claws. The fitted Ring01 camera preserves the CAD and separates
    # all four projected tips.
    isometric = _camera_with_azimuth(28.0, -5.0, 30.0)
    opposite = _camera_with_azimuth(28.0, -5.0, 210.0)
    definitions = (
        ("ISOMETRIC (FITTED) - 4 CLAWS", isometric),
        ("GEM FACE - 4 CLAWS", CAMERAS["front"]),
        ("HOOP PROFILE", CAMERAS["side"]),
        ("OPPOSITE - 4 CLAWS", opposite),
    )
    cells = []
    for label, camera in definitions:
        transform = _jewelry_transform(mesh, camera, (25, 25, 395, 395))
        cells.append(_clean_render(mesh, camera, (420, 420), transform, label))
    preview = np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))
    preview_path = output_dir / "ring01_phase3_3_2_visibility_preview.png"
    cv2.imwrite(str(preview_path), preview)

    cap_radius = float(build["parameters"]["prong_cap_radius"])
    visibility = {
        "isometric": _visibility_metrics(build["topology"]["prongs"], isometric, cap_radius),
        "opposite": _visibility_metrics(build["topology"]["prongs"], opposite, cap_radius),
        "gem_face": _visibility_metrics(build["topology"]["prongs"], CAMERAS["front"], cap_radius),
    }
    checks = {
        "four_prong_records": [record["prong_id"] for record in build["topology"]["prongs"]] == list(PRONG_IDS),
        "four_prong_topology": build["topology"]["prong_count"] == 4,
        "single_valid_metal_brep": build["validation"]["metal_solid_count"] == 1 and build["validation"]["metal_brep_valid"],
        "all_inspection_views_have_distinct_tips": all(item["four_distinct_tip_positions"] for item in visibility.values()),
        "oblique_views_avoid_center_axis_occlusion": visibility["isometric"]["no_tip_on_center_axis"] and visibility["opposite"]["no_tip_on_center_axis"],
        "phase331_visual_metrics_preserved": validation["mean_silhouette_iou"] >= 0.80,
    }
    report = {
        "stage": "phase3_3_2_prong_visibility_correction",
        "correction_type": "inspection_camera_and_visibility_validation",
        "geometry_changed": False,
        "geometry_reason": "The exact model already contains four symmetric claws; rotating them would regress the front and top references.",
        "source_step": str(step_path),
        "source_step_sha256": _sha256(step_path),
        "camera_correction": {
            "previous_isometric_azimuth_degrees": -45.0,
            "reference_tilt_from_stone_axis_degrees": 28.0,
            "reference_roll_degrees": -5.0,
            "isometric_azimuth_degrees": 30.0,
            "opposite_azimuth_degrees": 210.0,
        },
        "visibility": visibility,
        "checks": checks,
        "checkpoint_valid": all(checks.values()),
        "preserved_exact_metrics": {
            "mean_silhouette_iou": validation["mean_silhouette_iou"],
            "mean_detail_region_iou": validation["mean_detail_region_iou"],
            "mean_external_boundary_f1_2px": validation["mean_external_boundary_f1_2px"],
        },
        "preview": str(preview_path),
        "manufacturing_accuracy_validated": False,
    }
    (output_dir / "phase3_3_2_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/ring01_phase3_3_1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_3_2"))
    args = parser.parse_args()
    report = correct(args.source_dir, args.output_dir)
    print(json.dumps({"checkpoint_valid": report["checkpoint_valid"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
