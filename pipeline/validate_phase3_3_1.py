"""Strict, exported-CAD validation for Phase 3.3.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from phase3_objective import bbox, boundary_f1, crop, detail_regions, evaluate_regions, iou, read_component_masks
from reconstruct_phase3 import CAMERAS
from refine_phase3_1 import _camera, _jewelry_transform, _render
from refine_phase3_3 import VIEW_WEIGHTS, VISIBILITY, VIEWS, _camera_with_azimuth, _perspective_transform, _profile_camera, _render_perspective
from validate_phase3_2 import _basis_camera, _clean_render


def _fit_angled(mesh, target, frame, center_tilt=32, center_roll=22, center_azimuth=0, center_distance=60.0):
    best = (-1.0, center_tilt, center_roll, center_azimuth, center_distance, CAMERAS["angled"])
    distances = (center_distance,)
    for tilt in range(center_tilt - 4, center_tilt + 5, 4):
        for roll in range(center_roll - 4, center_roll + 5, 4):
            for azimuth in range(center_azimuth - 4, center_azimuth + 5, 4):
                camera = _camera_with_azimuth(tilt, roll, azimuth)
                for distance in distances:
                    transform = _perspective_transform(mesh, camera, frame, distance)
                    mask = _render_perspective(mesh, camera, target.shape, transform, distance)
                    intersection = np.count_nonzero((target > 0) & (mask > 0))
                    union = np.count_nonzero((target > 0) | (mask > 0))
                    score = intersection / max(1, union)
                    if score > best[0]:
                        best = (score, tilt, roll, azimuth, distance, camera)
    return (best[5], best[4]), {"tilt_degrees": best[1], "roll_degrees": best[2], "azimuth_degrees": best[3],
                                      "camera_distance_model_units": round(best[4], 4),
                                      "silhouette_iou": round(best[0], 6), "projection": "perspective", "calibrated": False}


def _fit_profile(mesh, target, frame, region, view, center_azimuth):
    best = (-1.0, center_azimuth, CAMERAS[view])
    for azimuth in range(center_azimuth - 4, center_azimuth + 5, 2):
        camera = _profile_camera(view, azimuth)
        transform = _jewelry_transform(mesh, camera, frame)
        rendered, _ = _render(mesh, camera, target.shape, transform)
        score = 0.5 * iou(target, rendered) + 0.5 * iou(crop(target, region), crop(rendered, region))
        if score > best[0]:
            best = (score, azimuth, camera)
    return best[2], {"azimuth_degrees": best[1], "silhouette_and_detail_score": round(best[0], 6), "calibrated": False}


def _overlay(image, target, rendered, title):
    result = image.copy()
    agreement = (target > 0) & (rendered > 0)
    missing = (target > 0) & (rendered == 0)
    excess = (target == 0) & (rendered > 0)
    result[agreement] = (result[agreement].astype(np.float32) * 0.45 + np.array([45, 180, 45]) * 0.55).astype(np.uint8)
    result[missing] = (25, 25, 225)
    result[excess] = (225, 125, 20)
    canvas = np.full((result.shape[0] + 25, result.shape[1], 3), 248, np.uint8)
    canvas[25:] = result
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (22, 22, 22), 1, cv2.LINE_AA)
    return canvas


def validate(input_dir: Path, masks_dir: Path, previous_dir: Path, phase_dir: Path) -> dict:
    build = json.loads((phase_dir / "phase3_3_1_build.json").read_text(encoding="utf-8"))
    fit = json.loads((phase_dir / "phase3_3_1_fit.json").read_text(encoding="utf-8"))
    masks_report = json.loads((masks_dir / "phase2_2_validation.json").read_text(encoding="utf-8"))
    targets = read_component_masks(masks_dir, "ring01", VIEWS)
    raw_jewelry_targets = {view: targets[view]["jewelry"].copy() for view in VIEWS}
    reviewed_dir = phase_dir / "evaluation_masks" / "jewelry"
    reviewed_masks_used = reviewed_dir.is_dir()
    if reviewed_masks_used:
        for view in VIEWS:
            path = reviewed_dir / f"ring01_{view}_jewelry.png"
            reviewed = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if reviewed is None:
                raise FileNotFoundError(path)
            targets[view]["jewelry"] = np.where(reviewed > 127, 255, 0).astype(np.uint8)
    frames = {view: bbox(targets[view]["jewelry"]) for view in VIEWS}
    regions = detail_regions(targets)
    meshes = {
        "jewelry": trimesh.load(phase_dir / "ring01_phase3_3_1.stl", force="mesh"),
        "shank": trimesh.load(phase_dir / "ring01_phase3_3_1_shank.stl", force="mesh"),
        "stone_amodal": trimesh.load(phase_dir / "ring01_phase3_3_1_stone.stl", force="mesh"),
        "setting": trimesh.load(phase_dir / "ring01_phase3_3_1_setting.stl", force="mesh"),
        "prongs": trimesh.load(phase_dir / "ring01_phase3_3_1_prongs.stl", force="mesh"),
    }
    previous = trimesh.load(previous_dir / "ring01_phase3_3.stl", force="mesh")
    cameras = dict(CAMERAS)
    proxy_camera = fit["camera_fit"]["angled"]
    cameras["angled"], camera_fit = _fit_angled(
        meshes["jewelry"], targets["angled"]["jewelry"], frames["angled"],
        int(proxy_camera["tilt_degrees"]), int(proxy_camera["roll_degrees"]), int(proxy_camera.get("azimuth_degrees", 0)),
        float(proxy_camera["camera_distance_model_units"]),
    )
    camera_fit["search"] = "27-pose local exported-mesh verification around cached proxy camera"
    profile_fits = {}
    for view in ("side", "back"):
        center = int(fit["camera_fit"]["profiles"][view]["azimuth_degrees"])
        cameras[view], profile_fits[view] = _fit_profile(
            meshes["jewelry"], targets[view]["jewelry"], frames[view], regions[view], view, center,
        )

    rendered, transforms = {}, {}
    for view in VIEWS:
        if isinstance(cameras[view], tuple):
            camera, distance = cameras[view]
            transforms[view] = _perspective_transform(meshes["jewelry"], camera, frames[view], distance)
        else:
            camera, distance = cameras[view], None
            transforms[view] = _jewelry_transform(meshes["jewelry"], camera, frames[view])
        rendered[view] = {}
        for component, mesh in meshes.items():
            if distance is None:
                mask, _ = _render(mesh, camera, targets[view][component].shape, transforms[view])
            else:
                mask = _render_perspective(mesh, camera, targets[view][component].shape, transforms[view], distance)
            rendered[view][component] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    objective, region_metrics = evaluate_regions(targets, rendered, regions, VISIBILITY, VIEW_WEIGHTS)

    views = {}
    comparisons = {}
    for view in VIEWS:
        target, output = targets[view]["jewelry"], rendered[view]["jewelry"]
        if isinstance(cameras[view], tuple):
            camera, distance = cameras[view]
        else:
            camera, distance = cameras[view], None
        global_boundary = boundary_f1(target, output, tolerance=2, external_only=True)
        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        if isinstance(cameras[view], tuple):
            previous_transform = _perspective_transform(previous, camera, frames[view], distance)
            previous_mask = _render_perspective(previous, camera, target.shape, previous_transform, distance)
        else:
            previous_transform = _jewelry_transform(previous, camera, frames[view])
            previous_mask, _ = _render(previous, camera, target.shape, previous_transform)
        old = _overlay(image, target, previous_mask, "PHASE 3.3 ERROR")
        new = _overlay(image, target, output, f"PHASE 3.3 IoU {region_metrics[view]['silhouette_iou']:.3f}")
        source = np.full((image.shape[0] + 25, image.shape[1], 3), 248, np.uint8)
        source[25:] = image
        cv2.putText(source, f"{view.upper()} REFERENCE", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (22, 22, 22), 1, cv2.LINE_AA)
        path = phase_dir / "comparisons" / f"ring01_{view}_reference_vs_phase3_3_1.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack([source, old, new]))
        comparisons[view] = str(path)
        mask_path = phase_dir / "renders" / f"ring01_{view}_silhouette.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), output)
        views[view] = dict(region_metrics[view]) | {"external_boundary_f1_2px": round(global_boundary, 6)}
        views[view]["raw_machine_mask_iou"] = round(iou(raw_jewelry_targets[view], output), 6)

    definitions = (
        ("ISOMETRIC", _basis_camera("isometric", (1.0, -1.0, 0.8))), ("GEM FACE", CAMERAS["front"]),
        ("HOOP PROFILE", CAMERAS["side"]), ("OPPOSITE", _basis_camera("opposite", (-1.0, -1.0, 0.55))),
    )
    cells = []
    for label, camera in definitions:
        transform = _jewelry_transform(meshes["jewelry"], camera, (25, 25, 395, 395))
        cells.append(_clean_render(meshes["jewelry"], camera, (420, 420), transform, label))
    preview = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    preview_path = phase_dir / "previews" / "ring01_phase3_3_1_clean_preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)

    mean_iou = float(np.mean([record["silhouette_iou"] for record in views.values()]))
    mean_raw_iou = float(np.mean([record["raw_machine_mask_iou"] for record in views.values()]))
    mean_detail = float(np.mean([record["detail_region_iou"] for record in views.values()]))
    mean_boundary = float(np.mean([record["external_boundary_f1_2px"] for record in views.values()]))
    principal_floor = min(views[v]["silhouette_iou"] for v in ("front", "side", "top", "back"))
    topology_checks = {
        "metal_single_closed_valid_brep": build["validation"]["metal_is_single_solid"] and build["validation"]["metal_brep_valid"],
        "stone_single_closed_valid_brep": build["validation"]["stone_is_single_solid"] and build["validation"]["stone_brep_valid"],
        "four_curved_prongs": build["topology"]["prong_count"] == 4 and build["topology"]["curved_prongs"],
        "four_stable_prong_ids": build["topology"].get("individual_prong_ids") == ["prong_ne", "prong_nw", "prong_sw", "prong_se"],
        "cubic_claw_geometry": build["topology"].get("prong_curve_version") == "cubic_claw_v1",
        "open_two_rail_gallery": build["topology"]["open_gallery"] and build["topology"]["gallery_rail_count"] == 2,
        "one_shank_four_shoulders": build["topology"]["one_shank"] and build["topology"]["cathedral_shoulder_count"] == 4,
        "faceted_separate_stone": build["topology"]["faceted_stone"] and build["validation"]["assembly_solid_count"] == 2,
        "stl_watertight": build["validation"]["mesh_watertight"],
    }
    exit_checks = {
        "mean_silhouette_iou_at_least_0_80": mean_iou >= 0.80,
        "mean_detail_region_iou_at_least_0_75": mean_detail >= 0.75,
        "mean_external_boundary_f1_at_least_0_75": mean_boundary >= 0.75,
        "principal_view_floor_at_least_0_70": principal_floor >= 0.70,
        "angled_iou_at_least_0_70": views["angled"]["silhouette_iou"] >= 0.70,
        "topology_and_solids_all_pass": all(topology_checks.values()),
        "human_ground_truth_masks_available": bool(masks_report.get("ground_truth_metrics_available", False)),
        "metric_scale_calibrated": False,
    }
    target_90_checks = {
        "mean_silhouette_iou_at_least_0_90": mean_iou >= 0.90,
        "every_principal_view_at_least_0_85": principal_floor >= 0.85,
        "angled_iou_at_least_0_82": views["angled"]["silhouette_iou"] >= 0.82,
        "mean_detail_iou_at_least_0_85": mean_detail >= 0.85,
        "mean_boundary_f1_at_least_0_88": mean_boundary >= 0.88,
        "topology_and_solids_all_pass": all(topology_checks.values()),
    }
    report = {
        "stage": "phase3_3_1_strict_exported_cad_validation", "checkpoint_valid": all(v for k, v in topology_checks.items() if k != "stl_watertight"),
        "phase3_exit_gate_passed": all(exit_checks.values()), "manufacturing_accuracy_validated": False,
        "balanced_objective": round(objective, 6), "mean_silhouette_iou": round(mean_iou, 6),
        "mean_raw_machine_mask_iou": round(mean_raw_iou, 6), "physics_corrected_masks_used": reviewed_masks_used,
        "mean_detail_region_iou": round(mean_detail, 6), "mean_external_boundary_f1_2px": round(mean_boundary, 6),
        "principal_view_floor": round(principal_floor, 6), "camera_fit": {"angled": camera_fit, "profiles": profile_fits},
        "topology_checks": topology_checks, "exit_checks": exit_checks, "target_90_checks": target_90_checks,
        "target_90_reached": all(target_90_checks.values()), "views": views,
        "comparisons": comparisons, "clean_preview": str(preview_path),
        "uncertainty": {"hidden_geometry": "inferred", "physical_scale": "uncalibrated",
                        "reference_masks": "physics-corrected machine masks, not human ground truth" if reviewed_masks_used else "machine-generated, not human ground truth",
                        "angled_camera": "estimated from silhouette"},
        "decision": "Remain in Phase 3 and refine; do not advance while any strict exit check fails.",
    }
    (phase_dir / "phase3_3_1_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--previous-dir", type=Path, default=Path("data/ring01_phase3_3"))
    parser.add_argument("--phase-dir", type=Path, default=Path("data/ring01_phase3_3_1"))
    args = parser.parse_args()
    report = validate(args.input_dir, args.masks_dir, args.previous_dir, args.phase_dir)
    print(json.dumps({k: report[k] for k in ("checkpoint_valid", "phase3_exit_gate_passed", "mean_silhouette_iou", "mean_detail_region_iou", "mean_external_boundary_f1_2px")}, indent=2))


if __name__ == "__main__":
    main()
