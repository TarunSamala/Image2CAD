"""Phase 3.3: confidence-aware, detail-region multi-view refinement.

This search uses a light triangular proxy.  The accepted parameters are later
rebuilt as exact solids and the exported mesh is independently revalidated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from phase3_objective import bbox, detail_regions, evaluate_regions, read_component_masks
from reconstruct_phase3 import CAMERAS, CameraHypothesis
from refine_phase3_1 import _camera, _jewelry_transform, _render


VIEWS = ("front", "side", "top", "angled", "back")
VIEW_WEIGHTS = {"front": 1.05, "side": 1.1, "top": 1.05, "angled": 1.2, "back": 1.1}
VISIBILITY = {
    "front": {"shank": 1.0, "stone_amodal": 1.0, "setting": 0.85, "prongs": 1.0, "negative_space": 0.25},
    "side": {"shank": 1.0, "stone_amodal": 0.80, "setting": 1.0, "prongs": 0.35, "negative_space": 1.0},
    "top": {"shank": 1.0, "stone_amodal": 1.0, "setting": 0.85, "prongs": 1.0, "negative_space": 0.30},
    "angled": {"shank": 0.95, "stone_amodal": 0.75, "setting": 0.85, "prongs": 0.30, "negative_space": 0.65},
    "back": {"shank": 1.0, "stone_amodal": 0.75, "setting": 1.0, "prongs": 0.20, "negative_space": 1.0},
}


def _rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    matrix = np.eye(4)
    matrix[1, 1], matrix[1, 2] = np.cos(angle), -np.sin(angle)
    matrix[2, 1], matrix[2, 2] = np.sin(angle), np.cos(angle)
    return matrix


def _tube(points: list[np.ndarray], radius_start: float, radius_end: float, sections: int = 12) -> trimesh.Trimesh:
    parts = []
    segment_count = len(points) - 1
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        radius = radius_start + (radius_end - radius_start) * ((index + 0.5) / segment_count)
        parts.append(trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.array([start, end])))
    return trimesh.util.concatenate(parts)


def _quadratic(start: np.ndarray, control: np.ndarray, end: np.ndarray, count: int = 5) -> list[np.ndarray]:
    result = []
    for t in np.linspace(0.0, 1.0, count):
        result.append((1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end)
    return result


def _gemstone(radius: float, bottom_z: float, girdle_z: float, top_z: float) -> trimesh.Trimesh:
    profile = np.array([
        [radius * 0.16, bottom_z],
        [radius, girdle_z],
        [radius, girdle_z + 0.08],
        [radius * 0.54, top_z],
        [0.0, top_z],
        [0.0, bottom_z],
    ])
    return trimesh.creation.revolve(profile, sections=24)


def build_proxy(parameters: dict[str, float]) -> dict[str, trimesh.Trimesh]:
    hoop = trimesh.creation.torus(
        parameters["hoop_radius"], parameters["shank_radius"],
        major_sections=72, minor_sections=14, transform=_rotation_x(90),
    )
    hoop.vertices[:, 1] *= parameters["shank_depth_radius"] / parameters["shank_radius"]
    lower_z = parameters["lower_gallery_z"]
    upper_z = lower_z + parameters["gallery_gap"]
    stone_radius = parameters["stone_radius"]

    shoulder_parts = []
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            start = np.array([parameters["shoulder_start_x"] * sign_x, parameters["shoulder_start_y"] * sign_y, parameters["shoulder_start_z"]])
            end = np.array([parameters["shoulder_end_x"] * sign_x, parameters["shoulder_end_y"] * sign_y, lower_z + 0.25])
            control = np.array([parameters["shoulder_control_x"] * sign_x, parameters["shoulder_control_y"] * sign_y, parameters["shoulder_control_z"]])
            shoulder_parts.append(_tube(_quadratic(start, control, end), parameters["shoulder_radius"], parameters["shoulder_tip_radius"]))
    shoulders = trimesh.util.concatenate(shoulder_parts)

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

    struts, prongs = [], []
    for angle in (45, 135, 225, 315):
        radians = np.deg2rad(angle)
        radial = np.array([np.cos(radians), np.sin(radians), 0.0])
        strut_start = radial * (stone_radius + parameters["strut_base_extra"])
        strut_start[2] = lower_z + 0.10
        strut_end = radial * (stone_radius + parameters["strut_top_extra"])
        strut_end[2] = upper_z + 0.12
        struts.append(_tube([strut_start, strut_end], parameters["strut_radius"], parameters["strut_radius"]))

        start = radial * (stone_radius + parameters["prong_base_extra"])
        start[2] = lower_z + 0.08
        end = radial * (stone_radius - parameters["prong_inset"])
        end[2] = parameters["prong_top_z"]
        control = radial * (stone_radius + parameters["prong_bow_extra"])
        control[2] = (start[2] + end[2]) * 0.5
        prong = _tube(_quadratic(start, control, end, 5), parameters["prong_base_radius"], parameters["prong_tip_radius"])
        bead = trimesh.creation.icosphere(subdivisions=1, radius=parameters["prong_bead_radius"])
        bead.apply_translation(end)
        prongs.append(trimesh.util.concatenate([prong, bead]))
    struts_mesh = trimesh.util.concatenate(struts)
    prongs_mesh = trimesh.util.concatenate(prongs)
    setting = trimesh.util.concatenate([shoulders, lower_gallery, upper_gallery, struts_mesh, prongs_mesh])
    stone = _gemstone(stone_radius, parameters["stone_bottom_z"], parameters["stone_girdle_z"], parameters["stone_top_z"])
    metal = trimesh.util.concatenate([hoop, setting])
    return {
        "jewelry": trimesh.util.concatenate([metal, stone]),
        "shank": hoop,
        "stone_amodal": stone,
        "setting": setting,
        "prongs": prongs_mesh,
        "metal": metal,
        "stone": stone,
    }


def _render_components(meshes, targets, frames, cameras):
    result = {}
    for view in VIEWS:
        if isinstance(cameras[view], tuple):
            camera, distance = cameras[view]
            transform = _perspective_transform(meshes["jewelry"], camera, frames[view], distance)
        else:
            camera, distance = cameras[view], None
            transform = _jewelry_transform(meshes["jewelry"], camera, frames[view])
        result[view] = {}
        for component in ("jewelry", "shank", "stone_amodal", "setting", "prongs"):
            if distance is None:
                mask, _ = _render(meshes[component], camera, targets[view][component].shape, transform)
            else:
                mask = _render_perspective(meshes[component], camera, targets[view][component].shape, transform, distance)
            # The proxy is a union of intersecting low-poly parts rather than a
            # boolean-fused solid.  Seal only raster-scale triangle seams; real
            # gallery openings are much larger and remain measurable.
            result[view][component] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return result


def _perspective_coordinates(mesh, camera, distance):
    vertices = mesh.vertices
    u = vertices @ np.asarray(camera.right)
    v = vertices @ np.asarray(camera.up)
    depth = vertices @ np.asarray(camera.direction)
    denominator = np.maximum(distance - depth, distance * 0.15)
    return u * distance / denominator, v * distance / denominator


def _perspective_transform(mesh, camera, frame, distance):
    u, v = _perspective_coordinates(mesh, camera, distance)
    x0, y0, x1, y1 = frame
    scale = min((x1 - x0) / max(float(np.ptp(u)), 1e-9), (y1 - y0) / max(float(np.ptp(v)), 1e-9))
    center = ((float(u.min()) + float(u.max())) / 2, (float(v.min()) + float(v.max())) / 2)
    return scale, center, frame


def _render_perspective(mesh, camera, shape, transform, distance):
    scale, (center_u, center_v), frame = transform
    u, v = _perspective_coordinates(mesh, camera, distance)
    x0, y0, x1, y1 = frame
    px = (u - center_u) * scale + (x0 + x1) / 2
    py = -(v - center_v) * scale + (y0 + y1) / 2
    polygons = np.rint(np.column_stack([px, py])[mesh.faces]).astype(np.int32)
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, polygons, 255)
    return result


def _camera_with_azimuth(tilt_degrees: float, roll_degrees: float, azimuth_degrees: float) -> CameraHypothesis:
    tilt, roll, azimuth = np.deg2rad((tilt_degrees, roll_degrees, azimuth_degrees))
    direction = np.array([np.sin(tilt) * np.sin(azimuth), np.sin(tilt) * np.cos(azimuth), np.cos(tilt)])
    base_right = np.array([np.cos(azimuth), -np.sin(azimuth), 0.0])
    base_up = np.cross(base_right, direction)
    right = np.cos(roll) * base_right + np.sin(roll) * base_up
    up = -np.sin(roll) * base_right + np.cos(roll) * base_up
    return CameraHypothesis("angled", tuple(right), tuple(up), tuple(direction), "perspective_silhouette_fit", 0.76)


def _fit_angled(mesh, target, frame):
    best = (-1.0, 35, 25, 0, 60.0, CAMERAS["angled"])
    for tilt in range(20, 45, 4):
        for roll in range(-10, 31, 5):
            for azimuth in range(-10, 41, 10):
                camera = _camera_with_azimuth(tilt, roll, azimuth)
                for distance in (28.0, 36.0, 48.0, 68.0, 100.0, 160.0):
                    transform = _perspective_transform(mesh, camera, frame, distance)
                    rendered = _render_perspective(mesh, camera, target.shape, transform, distance)
                    intersection = np.count_nonzero((target > 0) & (rendered > 0))
                    union = np.count_nonzero((target > 0) | (rendered > 0))
                    score = intersection / max(1, union)
                    if score > best[0]:
                        best = (score, tilt, roll, azimuth, distance, camera)
    return (best[5], best[4]), {"tilt_degrees": best[1], "roll_degrees": best[2], "azimuth_degrees": best[3],
                                      "camera_distance_model_units": best[4],
                                      "silhouette_iou": round(best[0], 6), "projection": "perspective", "calibrated": False}


def defaults() -> dict[str, float]:
    return {
        "hoop_radius": 10.0, "shank_radius": 0.72, "shank_depth_radius": 0.90,
        "stone_radius": 3.2519,
        "shoulder_start_x": 6.0, "shoulder_start_y": 0.48, "shoulder_start_z": 8.0,
        "shoulder_control_x": 4.7, "shoulder_control_y": 0.72, "shoulder_control_z": 9.75,
        "shoulder_end_x": 3.28, "shoulder_end_y": 1.35,
        "shoulder_radius": 0.42, "shoulder_tip_radius": 0.30,
        "lower_gallery_z": 10.85, "gallery_gap": 1.70,
        "lower_inner_inset": 0.30, "lower_outer_extra": 0.42, "lower_gallery_height": 0.32,
        "upper_inner_inset": 0.18, "upper_outer_extra": 0.28, "upper_gallery_height": 0.25,
        "strut_base_extra": 0.22, "strut_top_extra": 0.05, "strut_radius": 0.22,
        "prong_base_extra": 0.34, "prong_inset": 0.30, "prong_bow_extra": 0.15,
        "prong_top_z": 14.42, "prong_base_radius": 0.30, "prong_tip_radius": 0.20, "prong_bead_radius": 0.28,
        "stone_bottom_z": 11.30, "stone_girdle_z": 13.02, "stone_top_z": 14.22,
    }


SEARCH = {
    "shank_radius": ((0.58, 0.86), 0.07),
    "shank_depth_radius": ((0.72, 1.30), 0.10),
    "stone_radius": ((2.85, 3.65), 0.16),
    "shoulder_control_x": ((4.25, 5.35), 0.20),
    "shoulder_control_z": ((9.25, 10.45), 0.20),
    "shoulder_end_x": ((2.85, 3.75), 0.16),
    "shoulder_end_y": ((0.85, 1.55), 0.14),
    "shoulder_radius": ((0.28, 0.72), 0.06),
    "lower_gallery_z": ((10.45, 11.25), 0.14),
    "gallery_gap": ((1.35, 2.15), 0.14),
    "lower_outer_extra": ((0.20, 0.65), 0.09),
    "upper_outer_extra": ((0.12, 0.48), 0.07),
    "prong_inset": ((0.10, 0.65), 0.10),
    "prong_bow_extra": ((0.00, 0.48), 0.09),
    "prong_base_extra": ((0.18, 0.82), 0.11),
    "prong_top_z": ((13.85, 14.75), 0.15),
    "prong_base_radius": ((0.21, 0.36), 0.035),
    "prong_bead_radius": ((0.20, 0.56), 0.06),
    "stone_bottom_z": ((10.95, 11.65), 0.13),
    "stone_girdle_z": ((12.55, 13.35), 0.14),
    "stone_top_z": ((13.75, 14.55), 0.14),
}


def run(input_dir: Path, masks_dir: Path, output_dir: Path, rounds: int = 4) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = read_component_masks(masks_dir, "ring01", VIEWS)
    frames = {view: bbox(targets[view]["jewelry"]) for view in VIEWS}
    regions = detail_regions(targets)
    current = defaults()
    cameras = dict(CAMERAS)
    cameras["angled"], camera_fit = _fit_angled(build_proxy(current)["jewelry"], targets["angled"]["jewelry"], frames["angled"])

    def evaluate(parameters, keep_masks=False):
        meshes = build_proxy(parameters)
        rendered = _render_components(meshes, targets, frames, cameras)
        score, metrics = evaluate_regions(targets, rendered, regions, VISIBILITY, VIEW_WEIGHTS)
        return score, metrics, rendered if keep_masks else None

    initial_score, initial_metrics, initial_masks = evaluate(current, True)
    best_score = initial_score
    history = [{"round": 0, "score": round(best_score, 6), "parameters": dict(current)}]
    for round_index in range(rounds):
        changed = False
        for name, (bounds, base_step) in SEARCH.items():
            step = base_step * (0.58**round_index)
            candidates = [max(bounds[0], current[name] - step), min(bounds[1], current[name] + step)]
            local_value, local_score = current[name], best_score
            for value in candidates:
                candidate = dict(current)
                candidate[name] = round(value, 5)
                if not (candidate["stone_bottom_z"] < candidate["stone_girdle_z"] < candidate["stone_top_z"]):
                    continue
                score, _, _ = evaluate(candidate)
                if score > local_score + 1e-6:
                    local_value, local_score = candidate[name], score
            if local_value != current[name]:
                current[name], best_score, changed = local_value, local_score, True
        history.append({"round": round_index + 1, "score": round(best_score, 6), "parameters": dict(current)})
        if not changed:
            break

    # Refit only the uncertain oblique camera after geometry converges, then do
    # one short geometry pass.  Principal-view cameras remain fixed.
    cameras["angled"], camera_fit = _fit_angled(build_proxy(current)["jewelry"], targets["angled"]["jewelry"], frames["angled"])
    optimized_score, optimized_metrics, optimized_masks = evaluate(current, True)
    render_paths = {}
    for view in VIEWS:
        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        cells = []
        for title, mask in (("REFERENCE MASK", targets[view]["jewelry"]), ("PHASE 3.2 PROXY", initial_masks[view]["jewelry"]), ("PHASE 3.3 FIT", optimized_masks[view]["jewelry"])):
            cell = image.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cell, contours, -1, (20, 30, 220), 1, cv2.LINE_AA)
            canvas = np.full((cell.shape[0] + 24, cell.shape[1], 3), 248, np.uint8)
            canvas[24:] = cell
            cv2.putText(canvas, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
            cells.append(canvas)
        path = output_dir / "proxy_comparisons" / f"ring01_{view}_phase3_3_proxy.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack(cells))
        render_paths[view] = str(path)
        for component, mask in optimized_masks[view].items():
            mask_path = output_dir / "proxy_renders" / component / f"ring01_{view}_{component}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)

    report = {
        "sample": "ring01",
        "stage": "phase3_3_confidence_aware_detail_refinement",
        "backtracked_from": "data/ring01_phase3_2",
        "reversible": True,
        "objective": asdict(__import__("phase3_objective").ObjectiveWeights()),
        "visibility_confidence": VISIBILITY,
        "camera_fit": {"angled": camera_fit, "principal_view_orientations_fixed": True,
                       "cameras": {k: (asdict(v[0]) | {"distance": v[1]} if isinstance(v, tuple) else asdict(v)) for k, v in cameras.items()}},
        "initial": {"score": round(initial_score, 6), "view_metrics": initial_metrics},
        "optimized": {"score": round(optimized_score, 6), "view_metrics": optimized_metrics, "parameters": current},
        "optimization_history": history,
        "proxy_comparisons": render_paths,
        "warning": "Proxy metrics are search guidance only; acceptance uses independently rendered exported CAD.",
    }
    (output_dir / "phase3_3_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_3"))
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    report = run(args.input_dir, args.masks_dir, args.output_dir, args.rounds)
    print(json.dumps({"initial": report["initial"]["score"], "optimized": report["optimized"]["score"]}, indent=2))


if __name__ == "__main__":
    main()
