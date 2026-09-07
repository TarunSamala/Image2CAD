"""Experimental non-metric Phase 3 visual hulls for the five-view dataset."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import cv2
import numpy as np
import torch
import trimesh
from skimage import measure

from reconstruct_phase3 import CameraHypothesis
from refine_phase3_1 import _jewelry_transform
from train_dataset_phases import TinyJewelleryUNet, _binary_iou, _boundary_f1
from validate_phase3_2 import _basis_camera, _clean_render


VIEWS = ("front", "top", "iso", "lsv", "rsv")


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _load_model(path: Path, device: torch.device) -> tuple[TinyJewelleryUNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TinyJewelleryUNet(int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def _predict(model: TinyJewelleryUNet, image: np.ndarray, checkpoint: dict[str, Any], device: torch.device) -> np.ndarray:
    size = int(checkpoint["image_size"])
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return probability >= float(checkpoint["threshold"])


def _bbox_size(mask: np.ndarray) -> tuple[float, float]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise ValueError("Empty Phase 2 silhouette")
    _, _, width, height = cv2.boundingRect(points)
    return float(width), float(height)


def _warp_centered(mask: np.ndarray, scale: float, resolution: int) -> np.ndarray:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise ValueError("Empty Phase 2 silhouette")
    x, y, width, height = cv2.boundingRect(points)
    center_x, center_y = x + (width - 1) / 2, y + (height - 1) / 2
    matrix = np.asarray([
        [scale, 0.0, (resolution - 1) / 2 - scale * center_x],
        [0.0, scale, (resolution - 1) / 2 - scale * center_y],
    ])
    value = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (resolution, resolution),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    value[:2] = False
    value[-2:] = False
    value[:, :2] = False
    value[:, -2:] = False
    return value


def _calibrated_grids(masks: dict[str, np.ndarray], resolution: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    front_w, front_h = _bbox_size(masks["front"])
    top_w, top_h = _bbox_size(masks["top"])
    lsv_w, lsv_h = _bbox_size(masks["lsv"])
    rsv_w, rsv_h = _bbox_size(masks["rsv"])
    side_w, side_h = (lsv_w + rsv_w) / 2, (lsv_h + rsv_h) / 2
    # Solve isotropic per-view scales in log space so shared physical axes
    # agree: front/top share X, front/side share Z, top/side share Y.
    matrix = np.asarray([
        [1.0, -1.0, 0.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ])
    target = np.asarray([
        np.log(top_w / front_w),
        np.log(side_h / front_h),
        np.log(side_w / top_h),
        0.0,
    ])
    log_scales, residuals, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    relative = np.exp(log_scales)
    largest_extent = max(
        relative[0] * max(front_w, front_h),
        relative[1] * max(top_w, top_h),
        relative[2] * max(lsv_w, lsv_h, rsv_w, rsv_h),
    )
    common = (resolution - 8) / largest_extent
    scales = {
        "front": float(relative[0] * common),
        "top": float(relative[1] * common),
        "lsv": float(relative[2] * common),
        "rsv": float(relative[2] * common),
    }
    grids = {view: _warp_centered(masks[view], scales[view], resolution) for view in scales}
    return grids, {
        "method": "least_squares_shared_axis_scale_v1",
        "warp_scales_input_px_to_voxel": {key: round(value, 8) for key, value in scales.items()},
        "relative_view_scales": {
            "front": round(float(relative[0]), 8),
            "top": round(float(relative[1]), 8),
            "side": round(float(relative[2]), 8),
        },
        "log_constraint_residual": round(float(residuals[0]) if len(residuals) else 0.0, 8),
        "calibrated": False,
    }


def _side_candidates(lsv: np.ndarray, rsv: np.ndarray) -> dict[str, np.ndarray]:
    mirrored_rsv = np.fliplr(rsv)
    return {
        "intersection": lsv & mirrored_rsv,
        "union": lsv | mirrored_rsv,
        "left": lsv,
        "mirrored_right": mirrored_rsv,
    }


def _volume(front: np.ndarray, top: np.ndarray, side: np.ndarray) -> np.ndarray:
    # Axes are volume[z, y, x]. The prepared view semantics are treated as
    # orthographic hypotheses, not calibrated camera matrices.
    return front[:, None, :] & top[None, :, :] & side[:, :, None]


def _projections(volume: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "front": volume.any(axis=1),
        "top": volume.any(axis=0),
        "lsv": volume.any(axis=2),
        "rsv": np.fliplr(volume.any(axis=2)),
    }


def _fit_volume(masks: dict[str, np.ndarray], resolution: int) -> tuple[np.ndarray, str, dict[str, Any], dict[str, np.ndarray]]:
    grid, axis_alignment = _calibrated_grids(masks, resolution)
    candidates = []
    for mode, side in _side_candidates(grid["lsv"], grid["rsv"]).items():
        volume = _volume(grid["front"], grid["top"], side)
        if not np.any(volume):
            continue
        projected = _projections(volume)
        scores = {view: _binary_iou(grid[view], projected[view]) for view in ("front", "top", "lsv", "rsv")}
        candidates.append((float(np.mean(list(scores.values()))), mode, volume, scores))
    if not candidates:
        raise RuntimeError("No compatible volume remained after silhouette intersection")
    score, mode, volume, scores = max(candidates, key=lambda item: item[0])
    return volume, mode, {
        "selection_objective": round(score, 6),
        "candidate_objectives": {item[1]: round(item[0], 6) for item in candidates},
        "selected_projection_iou": {key: round(value, 6) for key, value in scores.items()},
        "axis_alignment": axis_alignment,
    }, grid


def _mesh_from_volume(volume: np.ndarray, nominal_size_mm: float) -> trimesh.Trimesh:
    padded = np.pad(volume.astype(np.float32), 1)
    vertices_zyx, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    vertices = vertices_zyx[:, [2, 1, 0]]
    vertices -= (np.asarray(padded.shape)[[2, 1, 0]] - 1.0) / 2.0
    vertices *= nominal_size_mm / max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def _write_3mf(mesh: trimesh.Trimesh, path: Path, title: str, nominal_size_mm: float) -> None:
    vertices = "".join(
        f'<vertex x="{x:.7f}" y="{y:.7f}" z="{z:.7f}"/>'
        for x, y, z in mesh.vertices
    )
    triangles = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>'
        for a, b, c in mesh.faces
    )
    warning = f"Non-metric visual hull; nominal display size {nominal_size_mm:g} mm; do not manufacture"
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f'<metadata name="Title">{escape(title)}</metadata>'
        f'<metadata name="Description">{escape(warning)}</metadata>'
        '<resources><object id="1" type="model"><mesh><vertices>'
        f'{vertices}</vertices><triangles>{triangles}</triangles></mesh></object></resources>'
        '<build><item objectid="1"/></build></model>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '</Types>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("3D/3dmodel.model", model)


def _cell(value: np.ndarray, label: str, width: int = 280, height: int = 230) -> np.ndarray:
    if value.ndim == 2:
        value = cv2.cvtColor(value.astype(np.uint8) * (255 if value.max() <= 1 else 1), cv2.COLOR_GRAY2BGR)
    scale = min(width / value.shape[1], height / value.shape[0])
    resized = cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 30, width, 3), 248, np.uint8)
    x, y = (width - resized.shape[1]) // 2, 30 + (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _error_overlay(target: np.ndarray, projected: np.ndarray) -> np.ndarray:
    canvas = np.full((*target.shape, 3), 245, np.uint8)
    canvas[target & projected] = (80, 180, 90)
    canvas[~target & projected] = (55, 65, 225)
    canvas[target & ~projected] = (225, 110, 45)
    return canvas


def _mesh_preview(mesh: trimesh.Trimesh, size: int = 512) -> np.ndarray:
    camera = _basis_camera("dataset_isometric", (1.0, -1.0, 0.8))
    frame = (40, 40, size - 40, size - 40)
    transform = _jewelry_transform(mesh, camera, frame)
    return _clean_render(mesh, camera, (size, size), transform, "ISOMETRIC VISUAL HULL")


def _comparison(
    record: dict[str, Any],
    masks: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    rows = []
    for view in ("front", "top", "lsv", "rsv"):
        image = _read(record["views"][view]["image_path"])
        target = masks[view]
        projected = projections[view]
        rows.append(np.hstack((
            _cell(image, f"{view.upper()} REFERENCE"),
            _cell(target, "PHASE 2 SILHOUETTE"),
            _cell(projected, f'PHASE 3 HULL IoU {metrics[view]["iou"]:.3f}'),
            _cell(_error_overlay(target, projected), "REPROJECTION ERROR"),
        )))
    iso_source = _read(record["views"]["iso"]["image_path"])
    mesh_preview = _mesh_preview(mesh)
    note = np.full((260, 560, 3), 248, np.uint8)
    cv2.putText(note, "NON-METRIC COARSE VISUAL HULL", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (65, 55, 180), 2, cv2.LINE_AA)
    cv2.putText(note, "No calibrated cameras or physical scale", (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(note, "STL / 3MF are research previews only", (18, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(note, "Not eligible for Phase 3.3 exact-CAD claims", (18, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    final_row = np.hstack((_cell(iso_source, "ISO REFERENCE", 280, 230), _cell(mesh_preview, "3D PREVIEW", 280, 230), note))
    body = np.vstack((*rows, final_row))
    header = np.full((88, body.shape[1], 3), 250, np.uint8)
    title = record["object_id"].replace("_", " ").upper()
    cv2.putText(header, f"{title} - DATASET PHASE 3 VISUAL HULL", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (72, 42, 12), 2, cv2.LINE_AA)
    cv2.putText(header, "Green agreement | red hull-only | blue silhouette-only", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (75, 75, 75), 1, cv2.LINE_AA)
    return np.vstack((header, body))


def _validate_3mf(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return set(("[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model")).issubset(archive.namelist())


def run(
    dataset_dir: Path,
    phase_run_dir: Path,
    output_dir: Path,
    resolution: int,
    nominal_size_mm: float,
    device_name: str,
    object_ids: set[str] | None,
) -> dict[str, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, checkpoint = _load_model(phase_run_dir / "checkpoints" / "whole_jewellery_unet.pt", device)
    records = [record for record in _manifest(dataset_dir) if not object_ids or record["object_id"] in object_ids]
    if not records:
        raise ValueError("No dataset objects matched --objects")
    object_reports = []
    split_metrics: dict[str, list[float]] = defaultdict(list)

    for record in records:
        object_id = record["object_id"]
        masks = {
            view: _predict(model, _read(record["views"][view]["image_path"]), checkpoint, device)
            for view in VIEWS
        }
        volume, side_mode, fit, calibrated_masks = _fit_volume(masks, resolution)
        mesh = _mesh_from_volume(volume, nominal_size_mm)
        projections = _projections(volume)
        metrics = {
            view: {
                "iou": round(_binary_iou(calibrated_masks[view], projections[view]), 6),
                "boundary_f1_2px": round(_boundary_f1(calibrated_masks[view], projections[view]), 6),
            }
            for view in projections
        }
        mean_iou = float(np.mean([item["iou"] for item in metrics.values()]))
        split_metrics[record["split"]].append(mean_iou)
        object_dir = output_dir / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        stl_path = object_dir / f"{object_id}_phase3_visual_hull_non_metric.stl"
        three_mf_path = object_dir / f"{object_id}_phase3_visual_hull_non_metric.3mf"
        preview_path = object_dir / f"{object_id}_phase3_comparison.png"
        mesh.export(stl_path)
        _write_3mf(mesh, three_mf_path, f"{object_id} Phase 3 visual hull", nominal_size_mm)
        comparison_masks = dict(masks)
        comparison_masks.update(calibrated_masks)
        cv2.imwrite(str(preview_path), _comparison(record, comparison_masks, projections, metrics, mesh))
        report = {
            "object_id": object_id,
            "split": record["split"],
            "stage": "dataset_phase3_visual_hull",
            "source_stage": "trained Phase 2 whole-jewellery silhouettes",
            "resolution": resolution,
            "selected_side_consensus": side_mode,
            "volume_voxels": int(np.count_nonzero(volume)),
            "mesh": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "body_count": int(mesh.body_count),
            },
            "reprojection": {"mean_iou": round(mean_iou, 6), "views": metrics, **fit},
            "exports": {"stl": str(stl_path), "3mf": str(three_mf_path), "preview": str(preview_path)},
            "export_checks": {"stl_readable": bool(trimesh.load(stl_path, force="mesh").vertices.size), "3mf_package_valid": _validate_3mf(three_mf_path)},
            "scale": {"calibrated": False, "nominal_display_size_mm": nominal_size_mm},
            "camera_calibrated": False,
            "phase3_3_eligible": False,
            "manufacturing_accuracy_validated": False,
            "warning": "Coarse visual hull from assumed orthographic views. Hidden concavities and component topology are unresolved.",
        }
        (object_dir / "phase3_visual_hull.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        object_reports.append(report)

    summary = {
        "stage": "dataset_phase3_visual_hull_batch",
        "object_count": len(object_reports),
        "device": str(device),
        "resolution": resolution,
        "exports": ["STL", "3MF"],
        "all_meshes_watertight": all(item["mesh"]["watertight"] for item in object_reports),
        "all_exports_readable": all(all(item["export_checks"].values()) for item in object_reports),
        "mean_reprojection_iou_by_split": {
            split: round(float(np.mean(values)), 6) for split, values in sorted(split_metrics.items())
        },
        "phase3_status": "experimental_non_metric_pass" if all(item["mesh"]["watertight"] for item in object_reports) else "failed",
        "phase3_3_status": "not_eligible_missing_semantic_components_cameras_scale_and_cad_targets",
        "manufacturing_accuracy_validated": False,
        "objects": object_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase3_batch_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/prepared_v1"))
    parser.add_argument("--phase-run-dir", type=Path, default=Path("dataset/phase_runs/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/phase_runs/v1/phase3_visual_hull"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--nominal-size-mm", type=float, default=30.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--objects", nargs="*", default=None, help="Optional object ids such as ring_007")
    args = parser.parse_args()
    report = run(
        args.dataset_dir,
        args.phase_run_dir,
        args.output_dir,
        args.resolution,
        args.nominal_size_mm,
        args.device,
        set(args.objects) if args.objects else None,
    )
    print(json.dumps({
        "object_count": report["object_count"],
        "all_meshes_watertight": report["all_meshes_watertight"],
        "all_exports_readable": report["all_exports_readable"],
        "mean_reprojection_iou_by_split": report["mean_reprojection_iou_by_split"],
        "phase3_status": report["phase3_status"],
        "phase3_3_status": report["phase3_3_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
