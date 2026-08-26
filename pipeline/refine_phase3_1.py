"""Phase 3.1: component-aware render-and-compare parameter fitting."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from reconstruct_phase3 import CAMERAS, CameraHypothesis


VIEWS = ("front", "side", "top", "angled", "back")
TARGET_COMPONENTS = ("jewelry", "shank", "stone_amodal", "setting", "prongs")
COMPONENT_WEIGHTS = {"jewelry": 0.46, "shank": 0.17, "stone_amodal": 0.22, "setting": 0.09, "prongs": 0.06}
VIEW_WEIGHTS = {"front": 1.1, "side": 1.0, "top": 1.1, "angled": 1.1, "back": 0.9}


def _read_mask(base: Path, view: str, component: str) -> np.ndarray:
    path = base / "masks" / component / f"ring01_{view}_{component}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("empty jewellery mask")
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width - 1, y + height - 1


def _rotation_x(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    matrix = np.eye(4)
    matrix[1, 1], matrix[1, 2] = np.cos(radians), -np.sin(radians)
    matrix[2, 1], matrix[2, 2] = np.sin(radians), np.cos(radians)
    return matrix


def _gemstone(radius: float, bottom_z: float, depth: float) -> trimesh.Trimesh:
    girdle_z = bottom_z + depth * 0.61
    top_z = bottom_z + depth
    profile = np.array([
        [0.0, bottom_z],
        [radius * 0.14, bottom_z],
        [radius, girdle_z],
        [radius, girdle_z + depth * 0.025],
        [radius * 0.55, top_z],
        [0.0, top_z],
        [0.0, bottom_z],
    ])
    return trimesh.creation.revolve(profile, sections=48)


def _frustum_between(start: np.ndarray, end: np.ndarray, radius_start: float, radius_end: float) -> trimesh.Trimesh:
    direction = end - start
    length = float(np.linalg.norm(direction))
    profile = np.array([[0.0, 0.0], [radius_start, 0.0], [radius_end, length], [0.0, length], [0.0, 0.0]])
    mesh = trimesh.creation.revolve(profile, sections=20)
    mesh.apply_transform(trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction / length))
    mesh.apply_translation(start)
    return mesh


def build_meshes(parameters: dict[str, float]) -> dict[str, trimesh.Trimesh]:
    radius = parameters["hoop_major_radius"]
    shank = trimesh.creation.torus(radius, parameters["shank_radius"], major_sections=72, minor_sections=18, transform=_rotation_x(90))
    stone_radius = parameters["stone_radius"]
    shoulder_end_x = parameters["shoulder_end_x"]
    shoulder_parts = []
    for sign in (-1.0, 1.0):
        shoulder_parts.append(trimesh.creation.cylinder(
            radius=parameters["shoulder_radius"],
            sections=24,
            segment=np.array([[6.0 * sign, 0.0, 8.0], [shoulder_end_x * sign, 0.0, parameters["seat_z"] + 0.3]]),
        ))
    shoulders = trimesh.util.concatenate(shoulder_parts)
    setting = trimesh.creation.annulus(
        max(0.7, stone_radius - parameters["seat_inner_inset"]),
        stone_radius + parameters["seat_outer_extra"],
        height=parameters["seat_height"],
        sections=56,
    )
    setting.apply_translation([0.0, 0.0, parameters["seat_z"] + parameters["seat_height"] / 2])
    prong_parts = []
    for angle in (45, 135, 225, 315):
        radians = np.deg2rad(angle)
        radial = stone_radius + parameters["prong_offset"]
        x, y = radial * np.cos(radians), radial * np.sin(radians)
        top_radial = radial - parameters["prong_inset"]
        tx, ty = top_radial * np.cos(radians), top_radial * np.sin(radians)
        prong_parts.append(_frustum_between(
            np.array([x, y, parameters["seat_z"] + 0.12]),
            np.array([tx, ty, parameters["seat_z"] + parameters["prong_height"]]),
            parameters["prong_radius"],
            parameters["prong_radius"] * 0.78,
        ))
    prongs = trimesh.util.concatenate(prong_parts)
    stone = _gemstone(stone_radius, parameters["stone_bottom_z"], parameters["stone_depth"])
    metal = trimesh.util.concatenate([shank, shoulders, setting, prongs])
    jewelry = trimesh.util.concatenate([metal, stone])
    return {
        "jewelry": jewelry,
        "shank": shank,
        "stone_amodal": stone,
        "setting": trimesh.util.concatenate([shoulders, setting, prongs]),
        "prongs": prongs,
        "metal": metal,
        "stone": stone,
    }


def _camera(tilt_degrees: float, roll_degrees: float) -> CameraHypothesis:
    tilt, roll = np.deg2rad(tilt_degrees), np.deg2rad(roll_degrees)
    direction = np.array([0.0, np.sin(tilt), np.cos(tilt)])
    base_right = np.array([1.0, 0.0, 0.0])
    base_up = np.array([0.0, -np.cos(tilt), np.sin(tilt)])
    right = np.cos(roll) * base_right + np.sin(roll) * base_up
    up = -np.sin(roll) * base_right + np.cos(roll) * base_up
    return CameraHypothesis("angled", tuple(right), tuple(up), tuple(direction), "semantic_silhouette_fit", 0.76)


def _projection(mesh: trimesh.Trimesh, camera: CameraHypothesis, frame: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    vertices = mesh.vertices
    u = vertices @ np.asarray(camera.right)
    v = vertices @ np.asarray(camera.up)
    x0, y0, x1, y1 = frame
    scale = min((x1 - x0) / max(float(np.ptp(u)), 1e-9), (y1 - y0) / max(float(np.ptp(v)), 1e-9))
    center_u, center_v = (float(u.min() + u.max()) / 2, float(v.min() + v.max()) / 2)
    px = (u - center_u) * scale + (x0 + x1) / 2
    py = -(v - center_v) * scale + (y0 + y1) / 2
    return px, py, scale, (center_u, center_v)


def _render(mesh: trimesh.Trimesh, camera: CameraHypothesis, shape: tuple[int, int], transform: tuple[float, tuple[float, float], tuple[int, int, int, int]] | None = None) -> tuple[np.ndarray, tuple[float, tuple[float, float], tuple[int, int, int, int]]]:
    if transform is None:
        raise ValueError("A jewellery projection transform is required")
    scale, (center_u, center_v), frame = transform
    vertices = mesh.vertices
    u = vertices @ np.asarray(camera.right)
    v = vertices @ np.asarray(camera.up)
    x0, y0, x1, y1 = frame
    px = (u - center_u) * scale + (x0 + x1) / 2
    py = -(v - center_v) * scale + (y0 + y1) / 2
    polygons = np.rint(np.column_stack([px, py])[mesh.faces]).astype(np.int32)
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, polygons, 255)
    return result, transform


def _jewelry_transform(mesh: trimesh.Trimesh, camera: CameraHypothesis, frame: tuple[int, int, int, int]) -> tuple[float, tuple[float, float], tuple[int, int, int, int]]:
    _, _, scale, center = _projection(mesh, camera, frame)
    return scale, center, frame


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero((first > 0) & (second > 0))
    union = np.count_nonzero((first > 0) | (second > 0))
    return float(intersection / max(1, union))


def _fit_angled_camera(mesh: trimesh.Trimesh, target: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[CameraHypothesis, dict[str, Any]]:
    best = (-1.0, 45, 0, CAMERAS["angled"])
    for tilt in range(20, 81, 5):
        for roll in range(-60, 61, 5):
            camera = _camera(tilt, roll)
            transform = _jewelry_transform(mesh, camera, frame)
            rendered, _ = _render(mesh, camera, target.shape, transform)
            score = _iou(target, rendered)
            if score > best[0]:
                best = (score, tilt, roll, camera)
    return best[3], {"tilt_degrees": best[1], "roll_degrees": best[2], "jewelry_iou": round(best[0], 5), "calibrated": False}


def evaluate(
    parameters: dict[str, float],
    targets: dict[str, dict[str, np.ndarray]],
    frames: dict[str, tuple[int, int, int, int]],
    cameras: dict[str, CameraHypothesis],
    return_masks: bool = False,
) -> tuple[float, dict[str, Any], dict[str, dict[str, np.ndarray]] | None]:
    meshes = build_meshes(parameters)
    weighted_total = 0.0
    weight_total = 0.0
    metrics = {}
    rendered_masks = {} if return_masks else None
    for view in VIEWS:
        transform = _jewelry_transform(meshes["jewelry"], cameras[view], frames[view])
        metrics[view] = {}
        if return_masks:
            rendered_masks[view] = {}
        for component in TARGET_COMPONENTS:
            rendered, _ = _render(meshes[component], cameras[view], targets[view][component].shape, transform)
            score = _iou(targets[view][component], rendered)
            metrics[view][component] = round(score, 5)
            component_weight = COMPONENT_WEIGHTS[component]
            if component == "prongs" and view not in ("front", "top", "angled"):
                component_weight *= 0.15
            weight = VIEW_WEIGHTS[view] * component_weight
            weighted_total += score * weight
            weight_total += weight
            if return_masks:
                rendered_masks[view][component] = rendered
    return weighted_total / weight_total, metrics, rendered_masks


def _comparison(image: np.ndarray, target: np.ndarray, rendered: np.ndarray, title: str) -> np.ndarray:
    overlay = image.copy()
    agreement = (target > 0) & (rendered > 0)
    missing = (target > 0) & (rendered == 0)
    excess = (target == 0) & (rendered > 0)
    overlay[agreement] = (overlay[agreement].astype(np.float32) * 0.4 + np.array([50, 190, 50]) * 0.6).astype(np.uint8)
    overlay[missing] = (20, 20, 230)
    overlay[excess] = (230, 120, 20)
    canvas = np.full((image.shape[0] + 26, image.shape[1], 3), 248, np.uint8)
    canvas[26:] = overlay
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def run(input_dir: Path, phase2_2_dir: Path, output_dir: Path, rounds: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {view: {component: _read_mask(phase2_2_dir, view, component) for component in TARGET_COMPONENTS} for view in VIEWS}
    frames = {view: _bbox(targets[view]["jewelry"]) for view in VIEWS}
    initial = {
        "hoop_major_radius": 10.0,
        "shank_radius": 0.90,
        "stone_radius": 3.375,
        "shoulder_end_x": 3.825,
        "shoulder_radius": 0.85,
        "seat_z": 10.65,
        "seat_inner_inset": 0.38,
        "seat_outer_extra": 1.05,
        "seat_height": 1.25,
        "prong_radius": 0.39,
        "prong_offset": 0.30,
        "prong_height": 4.40,
        "prong_inset": 0.28,
        "stone_bottom_z": 11.15,
        "stone_depth": 3.35,
    }
    bounds_steps = {
        "shank_radius": ((0.65, 1.55), 0.16),
        "stone_radius": ((2.70, 3.90), 0.20),
        "shoulder_end_x": ((3.10, 4.60), 0.22),
        "shoulder_radius": ((0.62, 1.25), 0.12),
        "seat_outer_extra": ((0.60, 1.45), 0.15),
        "seat_height": ((0.75, 1.75), 0.16),
        "prong_radius": ((0.26, 0.45), 0.04),
        "prong_offset": ((0.12, 0.50), 0.07),
        "prong_height": ((3.40, 5.30), 0.28),
        "stone_bottom_z": ((10.70, 11.70), 0.18),
        "stone_depth": ((2.80, 4.20), 0.22),
    }
    cameras = dict(CAMERAS)
    initial_meshes = build_meshes(initial)
    cameras["angled"], camera_fit = _fit_angled_camera(initial_meshes["jewelry"], targets["angled"]["jewelry"], frames["angled"])
    initial_score, initial_metrics, initial_masks = evaluate(initial, targets, frames, cameras, True)
    current, current_score = dict(initial), initial_score
    history = [{"iteration": 0, "score": round(initial_score, 6), "parameters": dict(initial)}]
    for round_index in range(rounds):
        changed = False
        for name, (bounds, base_step) in bounds_steps.items():
            step = base_step * (0.62 ** round_index)
            best_value, best_score = current[name], current_score
            for candidate_value in (max(bounds[0], current[name] - step), min(bounds[1], current[name] + step)):
                candidate = dict(current)
                candidate[name] = round(candidate_value, 5)
                score, _, _ = evaluate(candidate, targets, frames, cameras)
                if score > best_score + 1e-6:
                    best_value, best_score = candidate[name], score
            if best_value != current[name]:
                current[name], current_score, changed = best_value, best_score, True
        history.append({"iteration": round_index + 1, "score": round(current_score, 6), "parameters": dict(current)})
        if not changed:
            break
    optimized_score, optimized_metrics, optimized_masks = evaluate(current, targets, frames, cameras, True)

    comparisons = {}
    for view in VIEWS:
        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        before = _comparison(image, targets[view]["jewelry"], initial_masks[view]["jewelry"], f"{view.upper()} INITIAL IoU {initial_metrics[view]['jewelry']:.3f}")
        after = _comparison(image, targets[view]["jewelry"], optimized_masks[view]["jewelry"], f"{view.upper()} OPTIMIZED IoU {optimized_metrics[view]['jewelry']:.3f}")
        sheet = np.hstack([before, after])
        path = output_dir / "comparisons" / f"ring01_{view}_phase3_1_before_after.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet)
        comparisons[view] = str(path)
        for stage, stage_masks in (("initial", initial_masks), ("optimized", optimized_masks)):
            for component in TARGET_COMPONENTS:
                mask_path = output_dir / "renders" / stage / component / f"ring01_{view}_{component}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), stage_masks[view][component])

    report = {
        "sample": "ring01",
        "stage": "phase3_1_component_render_compare_refinement",
        "source_masks": str(phase2_2_dir / "phase2_2_components.json"),
        "objective": "weighted multi-view IoU over jewelry, shank, stone, setting, and observable prongs",
        "camera_fit": {"angled": camera_fit, "cameras": {view: asdict(camera) for view, camera in cameras.items()}},
        "initial": {"score": round(initial_score, 6), "parameters": initial, "view_component_iou": initial_metrics},
        "optimized": {"score": round(optimized_score, 6), "parameters": current, "view_component_iou": optimized_metrics},
        "improvement": round(optimized_score - initial_score, 6),
        "optimization_history": history,
        "comparisons": comparisons,
        "scale_status": "provisional; optimization fits normalized image shape, not measured millimetres",
    }
    (output_dir / "phase3_1_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--phase2-2-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_1"))
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    report = run(args.input_dir, args.phase2_2_dir, args.output_dir, args.rounds)
    print(json.dumps({"initial_score": report["initial"]["score"], "optimized_score": report["optimized"]["score"], "improvement": report["improvement"]}, indent=2))


if __name__ == "__main__":
    main()
