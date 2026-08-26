"""Render and validate the Phase 3.2 open-gallery CAD against Ring01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from reconstruct_phase3 import CAMERAS, CameraHypothesis
from refine_phase3_1 import _bbox, _camera, _fit_angled_camera, _iou, _jewelry_transform, _read_mask, _render


VIEWS = ("front", "side", "top", "angled", "back")


def _boundary_f1(target: np.ndarray, rendered: np.ndarray, tolerance: int = 2) -> float:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    target_edge = np.zeros_like(target)
    rendered_edge = np.zeros_like(rendered)
    target_contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rendered_contours, _ = cv2.findContours(rendered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(target_edge, target_contours, -1, 255, 1)
    cv2.drawContours(rendered_edge, rendered_contours, -1, 255, 1)
    target_near = cv2.dilate(target_edge, kernel, iterations=tolerance)
    rendered_near = cv2.dilate(rendered_edge, kernel, iterations=tolerance)
    precision = np.count_nonzero(cv2.bitwise_and(rendered_edge, target_near)) / max(1, np.count_nonzero(rendered_edge))
    recall = np.count_nonzero(cv2.bitwise_and(target_edge, rendered_near)) / max(1, np.count_nonzero(target_edge))
    return float(2 * precision * recall / max(1e-9, precision + recall))


def _head_holes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    x0, y0, x1, y1 = _bbox(mask)
    head_height = max(12, round((y1 - y0 + 1) * 0.36))
    roi = mask[y0 : min(mask.shape[0], y0 + head_height), x0 : x1 + 1]
    inverse = np.where(roi == 0, 1, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse)
    holes = np.zeros_like(mask)
    kept = 0
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        touches_border = x == 0 or y == 0 or x + width >= roi.shape[1] or y + height >= roi.shape[0]
        if not touches_border and area >= 4:
            local = np.where(labels == index, 255, 0).astype(np.uint8)
            holes[y0 : y0 + roi.shape[0], x0 : x0 + roi.shape[1]] = cv2.bitwise_or(
                holes[y0 : y0 + roi.shape[0], x0 : x0 + roi.shape[1]], local
            )
            kept += 1
    return holes, kept


def _clean_render(
    mesh: trimesh.Trimesh,
    camera: CameraHypothesis,
    shape: tuple[int, int],
    transform: tuple[float, tuple[float, float], tuple[int, int, int, int]],
    label: str | None = None,
) -> np.ndarray:
    supersample = 2
    scale, (center_u, center_v), frame = transform
    vertices = mesh.vertices
    right, up, direction = (np.asarray(camera.right), np.asarray(camera.up), np.asarray(camera.direction))
    u, v, depth = vertices @ right, vertices @ up, vertices @ direction
    x0, y0, x1, y1 = frame
    px = ((u - center_u) * scale + (x0 + x1) / 2) * supersample
    py = (-(v - center_v) * scale + (y0 + y1) / 2) * supersample
    polygons = np.rint(np.column_stack([px, py])[mesh.faces]).astype(np.int32)
    face_depth = depth[mesh.faces].mean(axis=1)
    light = np.asarray(camera.direction, np.float64) * 0.65 + np.array([0.25, -0.15, 0.55])
    light /= np.linalg.norm(light)
    shade = np.clip(0.25 + 0.75 * np.abs(mesh.face_normals @ light), 0.0, 1.0)
    label_height = 25 if label else 0
    canvas = np.full(((shape[0] + label_height) * supersample, shape[1] * supersample, 3), 248, np.uint8)
    y_offset = label_height * supersample
    polygons[:, :, 1] += y_offset
    for index in np.argsort(face_depth):
        value = int(72 + shade[index] * 160)
        color = (min(245, value + 10), value, max(45, value - 28))
        cv2.fillConvexPoly(canvas, polygons[index], color, cv2.LINE_AA)
    silhouette = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(silhouette, polygons, 255)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (75, 75, 75), 1 * supersample, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (7 * supersample, 18 * supersample), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * supersample, (20, 20, 20), 1 * supersample, cv2.LINE_AA)
    return cv2.resize(canvas, (shape[1], shape[0] + label_height), interpolation=cv2.INTER_AREA)


def _basis_camera(name: str, direction_values: tuple[float, float, float]) -> CameraHypothesis:
    direction = np.asarray(direction_values, np.float64)
    direction /= np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ world_up)) > 0.94:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    return CameraHypothesis(name, tuple(right), tuple(up), tuple(direction), "preview", 1.0)


def validate(input_dir: Path, phase1_dir: Path, phase2_2_dir: Path, phase3_1_dir: Path, phase3_2_dir: Path) -> dict:
    build = json.loads((phase3_2_dir / "phase3_2_build.json").read_text(encoding="utf-8"))
    previous = json.loads((phase3_1_dir / "phase3_1_validation.json").read_text(encoding="utf-8"))
    mesh = trimesh.load(phase3_2_dir / "ring01_phase3_2.stl", force="mesh")
    previous_mesh = trimesh.load(phase3_1_dir / "semantic" / "ring01_phase3_semantic.stl", force="mesh")
    targets = {view: _read_mask(phase2_2_dir, view, "jewelry") for view in VIEWS}
    frames = {view: _bbox(targets[view]) for view in VIEWS}
    cameras = dict(CAMERAS)
    cameras["angled"], angled_fit = _fit_angled_camera(mesh, targets["angled"], frames["angled"])
    best_camera, best_score = cameras["angled"], float(angled_fit["jewelry_iou"])
    coarse_tilt, coarse_roll = int(angled_fit["tilt_degrees"]), int(angled_fit["roll_degrees"])
    for tilt in range(coarse_tilt - 5, coarse_tilt + 6):
        for roll in range(coarse_roll - 5, coarse_roll + 6):
            candidate = _camera(tilt, roll)
            transform = _jewelry_transform(mesh, candidate, frames["angled"])
            rendered, _ = _render(mesh, candidate, targets["angled"].shape, transform)
            score = _iou(targets["angled"], rendered)
            if score > best_score:
                best_camera, best_score = candidate, score
                angled_fit["tilt_degrees"], angled_fit["roll_degrees"] = tilt, roll
    cameras["angled"] = best_camera
    angled_fit["jewelry_iou"] = round(best_score, 5)
    angled_fit["fine_search_step_degrees"] = 1

    views = {}
    for view in VIEWS:
        transform = _jewelry_transform(mesh, cameras[view], frames[view])
        rendered, _ = _render(mesh, cameras[view], targets[view].shape, transform)
        previous_transform = _jewelry_transform(previous_mesh, cameras[view], frames[view])
        previous_rendered, _ = _render(previous_mesh, cameras[view], targets[view].shape, previous_transform)
        phase1_mask = cv2.imread(str(phase1_dir / "masks" / f"ring01_{view}_mask.png"), cv2.IMREAD_GRAYSCALE)
        target_holes, target_hole_count = _head_holes(phase1_mask if phase1_mask is not None else targets[view])
        rendered_holes, rendered_hole_count = _head_holes(rendered)
        hole_iou = _iou(target_holes, rendered_holes) if np.any(target_holes) or np.any(rendered_holes) else None
        silhouette_iou = _iou(targets[view], rendered)
        boundary_f1 = _boundary_f1(targets[view], rendered)

        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(input_dir / f"ring01_{view}.png")
        source_cell = np.full((image.shape[0] + 25, image.shape[1], 3), 248, np.uint8)
        source_cell[25:] = image
        cv2.putText(source_cell, f"{view.upper()} REFERENCE", (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        old_cell = _clean_render(previous_mesh, cameras[view], image.shape[:2], previous_transform, "PHASE 3.1")
        new_cell = _clean_render(mesh, cameras[view], image.shape[:2], transform, f"PHASE 3.2 IoU {silhouette_iou:.3f}")
        sheet = np.hstack([source_cell, old_cell, new_cell])
        comparison_path = phase3_2_dir / "comparisons" / f"ring01_{view}_reference_vs_cad.png"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(comparison_path), sheet)
        mask_path = phase3_2_dir / "renders" / f"ring01_{view}_silhouette.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), rendered)
        views[view] = {
            "silhouette_iou": round(silhouette_iou, 5),
            "boundary_f1_tolerance_2px": round(boundary_f1, 5),
            "phase3_1_exported_iou": previous["exported_cad_jewelry_iou"][view],
            "target_head_hole_count": target_hole_count,
            "rendered_head_hole_count": rendered_hole_count,
            "head_hole_iou": round(hole_iou, 5) if hole_iou is not None else None,
            "comparison": str(comparison_path),
        }

    preview_definitions = (
        ("ISOMETRIC", _basis_camera("isometric", (1.0, -1.0, 0.8))),
        ("GEM FACE", CAMERAS["front"]),
        ("HOOP PROFILE", CAMERAS["side"]),
        ("OPPOSITE", _basis_camera("opposite", (-1.0, -1.0, 0.55))),
    )
    preview_cells = []
    preview_frame = (25, 25, 395, 395)
    for label, camera in preview_definitions:
        transform = _jewelry_transform(mesh, camera, preview_frame)
        preview_cells.append(_clean_render(mesh, camera, (420, 420), transform, label))
    preview = np.vstack([np.hstack(preview_cells[:2]), np.hstack(preview_cells[2:])])
    preview_path = phase3_2_dir / "previews" / "ring01_phase3_2_clean_preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)

    mean_iou = sum(record["silhouette_iou"] for record in views.values()) / len(views)
    mean_boundary = sum(record["boundary_f1_tolerance_2px"] for record in views.values()) / len(views)
    checks = {
        "metal_single_solid_pass": build["validation"]["metal_is_single_solid"],
        "stone_single_solid_pass": build["validation"]["stone_is_single_solid"],
        "open_gallery_topology_pass": build["topology"]["open_gallery"] and build["topology"]["gallery_rail_count"] == 2,
        "cathedral_topology_pass": build["topology"]["cathedral_shoulder_count"] == 4,
        "tapered_prongs_pass": build["topology"]["prongs_tapered_and_inward"] and build["topology"]["prong_count"] == 4,
        "faceted_stone_pass": build["topology"]["faceted_stone"],
        "view_specific_silhouette_floor_pass": (
            all(views[view]["silhouette_iou"] >= 0.55 for view in ("front", "side", "top", "back"))
            and views["angled"]["silhouette_iou"] >= 0.42
        ),
        "mean_boundary_pass": mean_boundary >= 0.45,
        "side_back_negative_space_pass": views["side"]["rendered_head_hole_count"] >= 1 and views["back"]["rendered_head_hole_count"] >= 1,
    }
    report = {
        "stage": "phase3_2_reference_and_topology_validation",
        "passed": all(checks.values()),
        "manufacturing_accuracy_validated": False,
        "checks": checks,
        "mean_silhouette_iou": round(mean_iou, 5),
        "mean_boundary_f1_tolerance_2px": round(mean_boundary, 5),
        "angled_camera_fit": angled_fit,
        "views": views,
        "clean_preview": str(preview_path),
        "remaining_gate": "Manual visual review, asymmetry refinement, and metric calibration are still required.",
    }
    (phase3_2_dir / "phase3_2_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--phase1-dir", type=Path, default=Path("data/ring01_phase1"))
    parser.add_argument("--phase2-2-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--phase3-1-dir", type=Path, default=Path("data/ring01_phase3_1"))
    parser.add_argument("--phase3-2-dir", type=Path, default=Path("data/ring01_phase3_2"))
    args = parser.parse_args()
    print(json.dumps(validate(args.input_dir, args.phase1_dir, args.phase2_2_dir, args.phase3_1_dir, args.phase3_2_dir), indent=2))


if __name__ == "__main__":
    main()
