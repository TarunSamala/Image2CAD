"""Phase 3.3.1: individual cubic-claw refinement from the Phase 3.3 checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from phase3_objective import bbox, detail_regions, evaluate_regions, iou, read_component_masks
from prong_geometry import PRONG_ANGLES, PRONG_IDS, claw_control, cubic_points, default_claw_parameters
from reconstruct_phase3 import CAMERAS
from refine_phase3_1 import _jewelry_transform, _render
from refine_phase3_3 import (
    VIEW_WEIGHTS,
    VIEWS,
    VISIBILITY,
    _camera_with_azimuth,
    _gemstone,
    _perspective_transform,
    _profile_camera,
    _render_perspective,
    _rotation_x,
    _tube,
)


SEARCH: dict[str, tuple[tuple[float, float], float]] = {}

# Phase 3.3.1 deliberately preserves the accepted Phase 3.3 projection.
# The previous unconstrained instance-mask search enlarged and detached the
# claws because reflective machine masks are not reliable geometric truth.



def _variable_tube(points: list[np.ndarray], radii: list[float], sections: int = 14) -> trimesh.Trimesh:
    parts = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        radius = (radii[index] + radii[index + 1]) * 0.5
        parts.append(trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.asarray([start, end])))
    return trimesh.util.concatenate(parts)


def _build_proxy(parameters: dict[str, Any]) -> dict[str, Any]:
    hoop = trimesh.creation.torus(
        parameters["hoop_radius"], parameters["shank_radius"], major_sections=72, minor_sections=14,
        transform=_rotation_x(90),
    )
    hoop.vertices[:, 1] *= parameters["shank_depth_radius"] / parameters["shank_radius"]
    lower_z = parameters["lower_gallery_z"]
    upper_z = lower_z + parameters["gallery_gap"]
    stone_radius = parameters["stone_radius"]

    shoulders = []
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            start = np.array([parameters["shoulder_start_x"] * sign_x, parameters["shoulder_start_y"] * sign_y, parameters["shoulder_start_z"]])
            control = np.array([parameters["shoulder_control_x"] * sign_x, parameters["shoulder_control_y"] * sign_y, parameters["shoulder_control_z"]])
            end = np.array([parameters["shoulder_end_x"] * sign_x, parameters["shoulder_end_y"] * sign_y, lower_z + 0.25])
            points = [(1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end for t in np.linspace(0, 1, 7)]
            shoulders.append(_tube(points, parameters["shoulder_radius"], parameters["shoulder_tip_radius"]))
    shoulders_mesh = trimesh.util.concatenate(shoulders)
    lower_gallery = trimesh.creation.annulus(
        r_min=max(0.6, stone_radius - parameters["lower_inner_inset"]),
        r_max=stone_radius + parameters["lower_outer_extra"], height=parameters["lower_gallery_height"], sections=40,
    )
    lower_gallery.apply_translation([0, 0, lower_z + parameters["lower_gallery_height"] / 2])
    upper_gallery = trimesh.creation.annulus(
        r_min=max(0.6, stone_radius - parameters["upper_inner_inset"]),
        r_max=stone_radius + parameters["upper_outer_extra"], height=parameters["upper_gallery_height"], sections=40,
    )
    upper_gallery.apply_translation([0, 0, upper_z + parameters["upper_gallery_height"] / 2])

    struts, prong_instances, prong_tips = [], {}, {}
    for angle in PRONG_ANGLES:
        radians = np.deg2rad(angle)
        radial = np.array([np.cos(radians), np.sin(radians), 0.0])
        start = radial * (stone_radius + parameters["strut_base_extra"]); start[2] = lower_z + 0.10
        end = radial * (stone_radius + parameters["strut_top_extra"]); end[2] = upper_z + 0.12
        struts.append(_tube([start, end], parameters["strut_radius"], parameters["strut_radius"]))
        control = claw_control(parameters, angle)
        points, radii = cubic_points(control)
        claw = _variable_tube(points, radii)
        cap = trimesh.creation.icosphere(subdivisions=2, radius=parameters["prong_cap_radius"])
        cap.apply_translation(control.tip)
        prong_instances[control.prong_id] = trimesh.util.concatenate([claw, cap])
        prong_tips[control.prong_id] = cap
    struts_mesh = trimesh.util.concatenate(struts)
    prongs_mesh = trimesh.util.concatenate(list(prong_instances.values()))
    setting = trimesh.util.concatenate([shoulders_mesh, lower_gallery, upper_gallery, struts_mesh, prongs_mesh])
    stone = _gemstone(stone_radius, parameters["stone_bottom_z"], parameters["stone_girdle_z"], parameters["stone_top_z"])
    metal = trimesh.util.concatenate([hoop, setting])
    return {
        "jewelry": trimesh.util.concatenate([metal, stone]), "shank": hoop, "stone_amodal": stone,
        "setting": setting, "prongs": prongs_mesh, "metal": metal, "stone": stone,
        "prong_instances": prong_instances, "prong_tips": prong_tips,
    }


def _extract_instances(mask: np.ndarray, stone_mask: np.ndarray) -> dict[str, np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    components = [(label, int(stats[label, cv2.CC_STAT_AREA]), centroids[label]) for label in range(1, count)]
    components = sorted((item for item in components if item[1] >= 20), key=lambda item: item[1], reverse=True)[:4]
    if len(components) != 4:
        raise ValueError(f"expected four visible prong instances, found {len(components)}")
    x, y, width, height = cv2.boundingRect(cv2.findNonZero(stone_mask))
    center_x, center_y = x + width / 2, y + height / 2
    result = {}
    for label, _, (cx, cy) in components:
        horizontal = "e" if cx >= center_x else "w"
        vertical = "n" if cy < center_y else "s"
        prong_id = f"prong_{vertical}{horizontal}"
        result[prong_id] = np.where(labels == label, 255, 0).astype(np.uint8)
    if set(result) != set(PRONG_IDS.values()):
        raise ValueError(f"ambiguous prong quadrants: {sorted(result)}")
    return result


def _cameras_from_checkpoint(fit: dict[str, Any]):
    cameras = dict(CAMERAS)
    angled = fit["camera_fit"]["angled"]
    cameras["angled"] = (
        _camera_with_azimuth(angled["tilt_degrees"], angled["roll_degrees"], angled["azimuth_degrees"]),
        angled["camera_distance_model_units"],
    )
    for view in ("side", "back"):
        cameras[view] = _profile_camera(view, fit["camera_fit"]["profiles"][view]["azimuth_degrees"])
    return cameras


def _render_all(meshes, targets, frames, cameras):
    rendered = {}
    for view in VIEWS:
        camera_entry = cameras[view]
        if isinstance(camera_entry, tuple):
            camera, distance = camera_entry
            transform = _perspective_transform(meshes["jewelry"], camera, frames[view], distance)
        else:
            camera, distance = camera_entry, None
            transform = _jewelry_transform(meshes["jewelry"], camera, frames[view])
        rendered[view] = {}
        for component in ("jewelry", "shank", "stone_amodal", "setting", "prongs"):
            if distance is None:
                mask, _ = _render(meshes[component], camera, targets[view][component].shape, transform)
            else:
                mask = _render_perspective(meshes[component], camera, targets[view][component].shape, transform, distance)
            rendered[view][component] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        rendered[view]["prong_instances"] = {}
        # Instance targets contain visible claw heads; occluded stems remain
        # supervised by the combined prong and profile masks.
        for prong_id, mesh in meshes["prong_tips"].items():
            if distance is None:
                mask, _ = _render(mesh, camera, targets[view]["jewelry"].shape, transform)
            else:
                mask = _render_perspective(mesh, camera, targets[view]["jewelry"].shape, transform, distance)
            rendered[view]["prong_instances"][prong_id] = mask
    return rendered


def run(previous_dir: Path, masks_dir: Path, input_dir: Path, output_dir: Path, rounds: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = json.loads((previous_dir / "phase3_3_fit.json").read_text(encoding="utf-8"))
    parameters = dict(prior["optimized"]["parameters"])
    parameters.update(default_claw_parameters())
    targets = read_component_masks(masks_dir, "ring01", VIEWS)
    reviewed_source = previous_dir / "evaluation_masks"
    if reviewed_source.is_dir():
        shutil.copytree(reviewed_source, output_dir / "evaluation_masks", dirs_exist_ok=True)
        for view in VIEWS:
            reviewed = cv2.imread(str(output_dir / "evaluation_masks" / "jewelry" / f"ring01_{view}_jewelry.png"), 0)
            targets[view]["jewelry"] = np.where(reviewed > 127, 255, 0).astype(np.uint8)
    frames = {view: bbox(targets[view]["jewelry"]) for view in VIEWS}
    regions = detail_regions(targets)
    target_instances = {view: _extract_instances(targets[view]["prongs"], targets[view]["stone_amodal"]) for view in ("front", "top")}
    cameras = _cameras_from_checkpoint(prior)

    def evaluate(candidate, keep=False):
        meshes = _build_proxy(candidate)
        rendered = _render_all(meshes, targets, frames, cameras)
        regional_score, metrics = evaluate_regions(targets, rendered, regions, VISIBILITY, VIEW_WEIGHTS)
        instance_scores = {
            view: {prong_id: iou(target_instances[view][prong_id], rendered[view]["prong_instances"][prong_id])
                   for prong_id in PRONG_IDS.values()}
            for view in ("front", "top")
        }
        instance_mean = float(np.mean([score for view in instance_scores.values() for score in view.values()]))
        combined = regional_score
        detail = {"regional_objective": regional_score, "instance_mean_iou": instance_mean,
                  "instance_iou": instance_scores, "views": metrics}
        return combined, detail, rendered if keep else None

    initial_score, initial_metrics, initial_rendered = evaluate(parameters, True)
    best_score = initial_score
    history = [{"round": 0, "score": round(best_score, 6), "parameters": dict(parameters)}]
    for round_index in range(rounds):
        changed = False
        for name, (bounds, base_step) in SEARCH.items():
            step = base_step * 0.58**round_index
            local_value, local_score = float(parameters[name]), best_score
            for value in (max(bounds[0], local_value - step), min(bounds[1], local_value + step)):
                candidate = dict(parameters); candidate[name] = round(value, 5)
                score, _, _ = evaluate(candidate)
                if score > local_score + 1e-6:
                    local_value, local_score = candidate[name], score
            if local_value != parameters[name]:
                parameters[name], best_score, changed = local_value, local_score, True
        history.append({"round": round_index + 1, "score": round(best_score, 6), "parameters": dict(parameters)})
        if not changed:
            break
    optimized_score, optimized_metrics, optimized_rendered = evaluate(parameters, True)

    instance_paths = {}
    for view in ("front", "top"):
        instance_paths[view] = {}
        for prong_id in PRONG_IDS.values():
            target_path = output_dir / "instance_masks" / "targets" / view / f"{prong_id}.png"
            render_path = output_dir / "instance_masks" / "renders" / view / f"{prong_id}.png"
            target_path.parent.mkdir(parents=True, exist_ok=True); render_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target_path), target_instances[view][prong_id])
            cv2.imwrite(str(render_path), optimized_rendered[view]["prong_instances"][prong_id])
            instance_paths[view][prong_id] = {"target": str(target_path), "render": str(render_path)}
    report = {
        "sample": "ring01", "stage": "phase3_3_1_individual_cubic_claw_refinement",
        "output_stage": "phase3_3_1", "reversible": True,
        "backtracked_from": str(previous_dir), "baseline_fit": str(previous_dir / "phase3_3_fit.json"),
        "prong_ids": list(PRONG_IDS.values()), "instance_supervision_views": ["front", "top"],
        "objective": {"regional": 1.0, "individual_prongs_diagnostic_only": 0.0},
        "selection_policy": "projection-preserving cubic conversion; machine prong instances are diagnostic until human-reviewed masks exist",
        "camera_fit": prior["camera_fit"],
        "initial": {"score": round(initial_score, 6), "metrics": initial_metrics},
        "optimized": {"score": round(optimized_score, 6), "metrics": optimized_metrics, "parameters": parameters},
        "optimization_history": history, "instance_masks": instance_paths,
        "warning": "Proxy scores guide search only; acceptance requires exported exact-solid validation.",
    }
    (output_dir / "phase3_3_1_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, default=Path("data/ring01_phase3_3"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_3_1"))
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    report = run(args.previous_dir, args.masks_dir, args.input_dir, args.output_dir, args.rounds)
    print(json.dumps({"initial": report["initial"]["score"], "optimized": report["optimized"]["score"],
                      "instance_iou": report["optimized"]["metrics"]["instance_mean_iou"]}, indent=2))


if __name__ == "__main__":
    main()



