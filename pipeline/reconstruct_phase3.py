"""Phase 3: coarse semantic 3D reconstruction from Phase 2.2 masks.

The Ring01 references are product renders rather than a calibrated camera
sequence.  This stage therefore uses explicit camera hypotheses and normalized
coordinates to build a visual hull.  It exports geometry and reprojections,
but does not claim metric scale or measured hidden surfaces.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from skimage.measure import marching_cubes


VIEWS = ("front", "side", "top", "angled", "back")


@dataclass(frozen=True)
class CameraHypothesis:
    name: str
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    direction: tuple[float, float, float]
    family: str
    confidence: float


CAMERAS = {
    # Front/top look approximately down the gemstone axis: image x/y map to
    # object x/y, while depth runs along object z.
    "front": CameraHypothesis("front", (1, 0, 0), (0, 1, 0), (0, 0, 1), "gemstone_face", 0.82),
    "top": CameraHypothesis("top", (1, 0, 0), (0, 1, 0), (0, 0, 1), "gemstone_face", 0.78),
    # Side/back expose the hoop plane: image x/z constrain object x/z.
    "side": CameraHypothesis("side", (1, 0, 0), (0, 0, 1), (0, 1, 0), "hoop_profile", 0.88),
    "back": CameraHypothesis("back", (-1, 0, 0), (0, 0, 1), (0, -1, 0), "hoop_profile", 0.84),
    # The angled view is a 45-degree interpolation between face and profile.
    "angled": CameraHypothesis(
        "angled",
        (1, 0, 0),
        (0, -0.70710678, 0.70710678),
        (0, 0.70710678, 0.70710678),
        "oblique",
        0.66,
    ),
}


def _read_mask(base: Path, view: str, component: str) -> np.ndarray:
    path = base / "masks" / component / f"ring01_{view}_{component}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _bbox(mask: np.ndarray, padding: int = 1) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Cannot derive a projection frame from an empty mask")
    x, y, width, height = cv2.boundingRect(points)
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(mask.shape[1] - 1, x + width - 1 + padding),
        min(mask.shape[0] - 1, y + height - 1 + padding),
    )


def _project(points: np.ndarray, camera: CameraHypothesis, frame: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    right = np.asarray(camera.right, np.float32)
    up = np.asarray(camera.up, np.float32)
    u = points @ right
    v = points @ up
    x0, y0, x1, y1 = frame
    px = np.rint(x0 + (u + 1.0) * 0.5 * (x1 - x0)).astype(np.int32)
    py = np.rint(y1 - (v + 1.0) * 0.5 * (y1 - y0)).astype(np.int32)
    return px, py


def _sample(mask: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    inside = (px >= 0) & (px < mask.shape[1]) & (py >= 0) & (py < mask.shape[0])
    result = np.zeros(px.shape, dtype=bool)
    result[inside] = mask[py[inside], px[inside]] > 0
    return result


def _voxel_points(resolution: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([value.ravel() for value in grid])


def _carve(
    points: np.ndarray,
    masks: dict[str, np.ndarray],
    frames: dict[str, tuple[int, int, int, int]],
    min_support: int,
    cameras: dict[str, CameraHypothesis],
) -> tuple[np.ndarray, dict[str, int]]:
    support = np.zeros(len(points), np.uint8)
    per_view = {}
    for view in VIEWS:
        px, py = _project(points, cameras[view], frames[view])
        hit = _sample(masks[view], px, py)
        support += hit.astype(np.uint8)
        per_view[view] = int(np.count_nonzero(hit))
    return support >= min_support, per_view


def _carve_camera_families(
    points: np.ndarray,
    masks: dict[str, np.ndarray],
    frames: dict[str, tuple[int, int, int, int]],
    cameras: dict[str, CameraHypothesis],
) -> tuple[np.ndarray, dict[str, int]]:
    """Require evidence from each independent camera family.

    Front/top and side/back are near-duplicate product views.  Treating all
    five as equal votes allows those duplicates to overrule the sole oblique
    view.  Family constraints avoid that bias.
    """
    hits = {}
    counts = {}
    for view in VIEWS:
        px, py = _project(points, cameras[view], frames[view])
        hits[view] = _sample(masks[view], px, py)
        counts[view] = int(np.count_nonzero(hits[view]))
    face = hits["front"] | hits["top"]
    profile = hits["side"] | hits["back"]
    return face & profile & hits["angled"], counts


def _mesh_from_volume(volume: np.ndarray) -> trimesh.Trimesh:
    if np.count_nonzero(volume) < 8:
        raise RuntimeError("Visual hull is empty; camera hypotheses and masks are inconsistent")
    spacing = 2.0 / (volume.shape[0] - 1)
    # A zero border ensures marching cubes closes surfaces that meet the
    # normalized reconstruction boundary.
    padded = np.pad(volume.astype(np.float32), 1, mode="constant")
    vertices, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(spacing, spacing, spacing))
    vertices -= 1.0 + spacing
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def _rasterize(points: np.ndarray, camera: CameraHypothesis, frame: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    px, py = _project(points, camera, frame)
    valid = (px >= 0) & (px < shape[1]) & (py >= 0) & (py < shape[0])
    result = np.zeros(shape, np.uint8)
    result[py[valid], px[valid]] = 255
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    return result


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero((first > 0) & (second > 0))
    union = np.count_nonzero((first > 0) | (second > 0))
    return float(intersection / max(1, union))


def _optimize_angled_camera(
    points: np.ndarray,
    reference: np.ndarray,
    frame: tuple[int, int, int, int],
) -> tuple[CameraHypothesis, dict[str, Any]]:
    """Estimate a constrained oblique pose from silhouette agreement.

    This is an orientation search, not camera calibration: focal length and
    metric translation remain unknown.
    """
    best_camera = CAMERAS["angled"]
    best_iou = -1.0
    best_angles = (45, 0)
    for tilt_degrees in range(15, 81, 5):
        tilt = np.deg2rad(tilt_degrees)
        direction = np.array([0.0, np.sin(tilt), np.cos(tilt)], np.float64)
        base_right = np.array([1.0, 0.0, 0.0], np.float64)
        base_up = np.array([0.0, -np.cos(tilt), np.sin(tilt)], np.float64)
        for roll_degrees in range(-60, 61, 5):
            roll = np.deg2rad(roll_degrees)
            right = np.cos(roll) * base_right + np.sin(roll) * base_up
            up = -np.sin(roll) * base_right + np.cos(roll) * base_up
            camera = CameraHypothesis(
                "angled",
                tuple(float(value) for value in right),
                tuple(float(value) for value in up),
                tuple(float(value) for value in direction),
                "oblique_silhouette_fit",
                0.72,
            )
            rendered = _rasterize(points, camera, frame, reference.shape)
            score = _mask_iou(reference, rendered)
            if score > best_iou:
                best_iou = score
                best_camera = camera
                best_angles = (tilt_degrees, roll_degrees)
    return best_camera, {
        "method": "constrained_orthographic_silhouette_search",
        "tilt_degrees": best_angles[0],
        "roll_degrees": best_angles[1],
        "initial_iou": round(_mask_iou(reference, _rasterize(points, CAMERAS["angled"], frame, reference.shape)), 5),
        "optimized_iou": round(best_iou, 5),
        "calibrated": False,
    }


def _comparison(reference: np.ndarray, rendered: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    intersection = np.count_nonzero((reference > 0) & (rendered > 0))
    union = np.count_nonzero((reference > 0) | (rendered > 0))
    reference_area = np.count_nonzero(reference)
    rendered_area = np.count_nonzero(rendered)
    metrics = {
        "silhouette_iou": round(intersection / max(1, union), 5),
        "silhouette_precision": round(intersection / max(1, rendered_area), 5),
        "silhouette_recall": round(intersection / max(1, reference_area), 5),
    }
    overlay = image.copy()
    reference_only = (reference > 0) & (rendered == 0)
    rendered_only = (rendered > 0) & (reference == 0)
    agreement = (reference > 0) & (rendered > 0)
    overlay[agreement] = (overlay[agreement].astype(np.float32) * 0.45 + np.array([50, 190, 50]) * 0.55).astype(np.uint8)
    overlay[reference_only] = (20, 20, 230)
    overlay[rendered_only] = (230, 120, 20)
    return overlay, metrics


def _export_mesh(mesh: trimesh.Trimesh, output_dir: Path, name: str) -> dict[str, str]:
    paths = {}
    for extension in ("stl", "obj", "glb"):
        path = output_dir / "meshes" / f"{name}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        paths[extension] = str(path)
    return paths


def _preview_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ world_up)) > 0.94:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    return right, up


def _render_preview(mesh: trimesh.Trimesh, output_dir: Path) -> str:
    """Create a dependency-free four-view surface preview for visual review."""
    np.random.seed(7)
    points, face_ids = trimesh.sample.sample_surface(mesh, 220000)
    normals = mesh.face_normals[face_ids]
    definitions = (
        ("ISOMETRIC", np.array([1.0, -1.0, 0.8])),
        ("GEM FACE", np.array([0.0, 0.0, 1.0])),
        ("HOOP PROFILE", np.array([0.0, 1.0, 0.0])),
        ("OPPOSITE", np.array([-1.0, -1.0, 0.55])),
    )
    cells = []
    for label, direction in definitions:
        direction = direction / np.linalg.norm(direction)
        right, up = _preview_basis(direction)
        u, v, depth = points @ right, points @ up, points @ direction
        span = max(float(np.ptp(u)), float(np.ptp(v)), 1e-9)
        size, margin, label_height = 420, 22, 34
        px = np.rint((u - (u.min() + u.max()) / 2) / span * (size - 2 * margin) + size / 2).astype(np.int32)
        py = np.rint(-(v - (v.min() + v.max()) / 2) / span * (size - 2 * margin) + size / 2 + label_height / 2).astype(np.int32)
        light = np.array([0.35, -0.45, 0.82])
        light /= np.linalg.norm(light)
        shade = np.clip(0.30 + 0.70 * np.abs(normals @ light), 0.0, 1.0)
        order = np.argsort(depth)
        canvas = np.full((size + label_height, size, 3), 245, np.uint8)
        for index in order:
            x, y = px[index], py[index]
            if 1 <= x < size - 1 and label_height <= y < size + label_height - 1:
                value = int(45 + shade[index] * 175)
                canvas[y, x] = (min(255, value + 20), value, max(20, value - 35))
        canvas[label_height:] = cv2.morphologyEx(canvas[label_height:], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cv2.putText(canvas, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        cells.append(canvas)
    sheet = np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))
    path = output_dir / "previews" / "ring01_phase3_mesh_preview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return str(path)


def run(
    input_dir: Path,
    phase2_2_dir: Path,
    output_dir: Path,
    resolution: int,
    min_support: int,
    reference_width_mm: float | None,
) -> dict[str, Any]:
    if resolution < 48:
        raise ValueError("resolution must be at least 48")
    if not 3 <= min_support <= len(VIEWS):
        raise ValueError("min_support must be between 3 and 5")
    output_dir.mkdir(parents=True, exist_ok=True)
    jewelry_masks = {view: _read_mask(phase2_2_dir, view, "jewelry") for view in VIEWS}
    stone_masks = {view: _read_mask(phase2_2_dir, view, "stone_amodal") for view in VIEWS}
    frames = {view: _bbox(jewelry_masks[view], padding=1) for view in VIEWS}
    points = _voxel_points(resolution)

    cameras = dict(CAMERAS)
    initial_occupied, _ = _carve(points, jewelry_masks, frames, min_support, cameras)
    cameras["angled"], angled_pose_fit = _optimize_angled_camera(
        points[initial_occupied], jewelry_masks["angled"], frames["angled"]
    )
    occupied, raw_support = _carve_camera_families(points, jewelry_masks, frames, cameras)
    volume = occupied.reshape((resolution, resolution, resolution))
    volume = cv2.morphologyEx(
        volume.astype(np.uint8).reshape(resolution, resolution * resolution),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ).reshape(volume.shape).astype(bool)
    mesh = _mesh_from_volume(volume)

    metric_scale = reference_width_mm is not None
    scale = 1.0
    if reference_width_mm is not None:
        current_width = float(mesh.bounds[1, 0] - mesh.bounds[0, 0])
        scale = reference_width_mm / max(current_width, 1e-9)
        mesh.apply_scale(scale)
    mesh_paths = _export_mesh(mesh, output_dir, "ring01_phase3_visual_hull")
    preview_path = _render_preview(mesh, output_dir)

    # Stone evidence is exported as a semantic point cloud.  Requiring three
    # views avoids forcing uncertain side/back amodal masks into a final solid.
    stone_occupied, stone_support = _carve(points, stone_masks, frames, 3, cameras)
    stone_points = points[stone_occupied] * scale
    stone_cloud = trimesh.points.PointCloud(stone_points, colors=[245, 210, 45, 255])
    stone_path = output_dir / "meshes" / "ring01_phase3_stone_evidence.ply"
    stone_cloud.export(stone_path)

    occupied_points = points[volume.ravel()]
    reprojections = {}
    for view in VIEWS:
        rendered = _rasterize(occupied_points, cameras[view], frames[view], jewelry_masks[view].shape)
        rendered_path = output_dir / "reprojections" / f"ring01_{view}_phase3_silhouette.png"
        rendered_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rendered_path), rendered)
        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(input_dir / f"ring01_{view}.png")
        overlay, metrics = _comparison(jewelry_masks[view], rendered, image)
        overlay_path = output_dir / "comparisons" / f"ring01_{view}_phase3_compare.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)
        reprojections[view] = {
            **metrics,
            "camera": cameras[view].__dict__,
            "projection_frame_px": list(frames[view]),
            "rendered_mask": str(rendered_path),
            "comparison": str(overlay_path),
        }

    report = {
        "sample": "ring01",
        "stage": "phase3_coarse_multiview_visual_hull",
        "source_stage": str(phase2_2_dir / "phase2_2_components.json"),
        "reversible": True,
        "coordinate_system": "normalized object coordinates" if not metric_scale else "millimetres from supplied reference width",
        "metric_scale": metric_scale,
        "reference_width_mm": reference_width_mm,
        "camera_source": "explicit hypotheses; not calibrated or COLMAP-recovered",
        "camera_warning": "Reference labels are product-view semantics, not verified orthogonal camera poses.",
        "angled_pose_fit": angled_pose_fit,
        "algorithm": {
            "method": "silhouette visual hull with independent camera-family constraints",
            "resolution": resolution,
            "initial_pose_search_view_support": min_support,
            "required_camera_families": ["front_or_top", "side_or_back", "angled"],
            "views": list(VIEWS),
        },
        "geometry": {
            "occupied_voxels": int(np.count_nonzero(volume)),
            "occupancy_fraction": round(np.count_nonzero(volume) / volume.size, 7),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "body_count": int(mesh.body_count),
            "bounds": np.round(mesh.bounds, 6).tolist(),
            "extents": np.round(mesh.extents, 6).tolist(),
            "volume": round(float(abs(mesh.volume)), 7),
        },
        "semantic_evidence": {
            "stone_point_count": int(len(stone_points)),
            "stone_point_cloud": str(stone_path),
            "status": "evidence_only_pending_parametric_gemstone_fit",
        },
        "raw_projection_support": raw_support,
        "raw_stone_support": stone_support,
        "mesh_artifacts": mesh_paths,
        "mesh_preview": preview_path,
        "reprojections": reprojections,
        "limitations": [
            "hidden surfaces are a visual-hull inference",
            "camera intrinsics and extrinsics are not calibrated",
            "normalized output has no manufacturing scale until a known dimension is supplied",
            "concavities invisible in silhouettes cannot be recovered",
            "stone evidence is not yet a faceted parametric gemstone",
        ],
    }
    (output_dir / "phase3_reconstruction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--phase2-2-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3"))
    parser.add_argument("--resolution", type=int, default=112)
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--reference-width-mm", type=float)
    args = parser.parse_args()
    report = run(args.input_dir, args.phase2_2_dir, args.output_dir, args.resolution, args.min_support, args.reference_width_mm)
    print(json.dumps({"output": str(args.output_dir), "geometry": report["geometry"], "metric_scale": report["metric_scale"]}, indent=2))


if __name__ == "__main__":
    main()
