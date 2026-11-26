"""Measure bounded depth relief against the STL-1 visual hull on train and val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from pipeline.audit_saraswati_blind_depth import _edge_agreement
from pipeline.benchmark_stl1_paired_meshes import _normal_metrics
from pipeline.infer_metric_depth_backend import load_depth_anything_v2
from pipeline.reconstruct_dataset_phase3 import (
    _fit_volume, _load_model, _manifest, _mesh_from_volume, _predict, _projections, _read,
)
from pipeline.refine_saraswati_blind_relief import _normalized_depth, _warp_centered_values
from pipeline.refine_saraswati_multiview_relief import carve_axis_relief
from pipeline.run_saraswati_blind_benchmark import (
    _align, _load_mesh, _normalize, _sha256, _surface_metrics, _write_json,
)
from pipeline.train_dataset_phases import _binary_iou


PHASE_ROOT = Path("dataset/phase_runs/v1")
VIEWS = ("front", "top", "lsv", "rsv")
EDGE_GATE = 0.35
STRENGTH = 0.06
DIRECTION = "high"  # Fixed from the existing upload relief proposal.


def _voxel_mesh(volume: np.ndarray) -> trimesh.Trimesh:
    # Cancel per-mesh display scaling so both meshes retain the same voxel frame.
    spans = []
    for axis in range(3):
        indices = np.flatnonzero(volume.any(axis=tuple(i for i in range(3) if i != axis)))
        spans.append(int(indices[-1] - indices[0] + 1))
    return _mesh_from_volume(volume, nominal_size_mm=float(max(spans)))


def _compare(base, refined, target, count):
    base_points, base_faces = trimesh.sample.sample_surface(base, count, seed=301)
    target_points, target_faces = trimesh.sample.sample_surface(target, count, seed=420)
    _, center, scale = _normalize(np.asarray(base.vertices))
    _, target_center, target_scale = _normalize(np.asarray(target.vertices))
    normalized = (base_points - center) / scale
    target_points = (target_points - target_center) / target_scale
    aligned, fit = _align(normalized, target_points)
    rotation = np.asarray(fit["rotation_source_to_target"])
    translation = (aligned - normalized @ rotation.T).mean(axis=0)
    result = {}
    surfaces = {"baseline": aligned}
    for name, mesh in (("baseline", base), ("depth_relief", refined)):
        if name == "baseline" or mesh is base:
            points, faces = aligned, base_faces
        else:
            points, faces = trimesh.sample.sample_surface(mesh, count, seed=301)
            points = ((points - center) / scale) @ rotation.T + translation
        surfaces[name] = points
        result[name] = _surface_metrics(points, target_points)
        result[name]["normal_consistency"] = _normal_metrics(
            points, mesh.face_normals[faces] @ rotation.T,
            target_points, target.face_normals[target_faces],
        )
    fit["translation_source_to_target"] = translation.tolist()
    fit["voxel_extent"] = scale
    fit["target_extent_in_obj_units"] = target_scale
    return result, fit, {**surfaces, "paired_obj": target_points}


def _preview(surfaces, path):
    # Shared viewport and colors make the three projections directly comparable.
    panels = []
    for x, y, axes in ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")):
        row = []
        for label, points in surfaces.items():
            panel = np.full((250, 260, 3), 245, dtype=np.uint8)
            pixels = np.rint(points[:, [x, y]] * [185, -185] + [130, 136]).astype(int)
            keep = ((pixels >= 0) & (pixels < [260, 250])).all(axis=1)
            panel[pixels[keep, 1], pixels[keep, 0]] = (166, 84, 42) if label != "paired_obj" else (45, 120, 186)
            cv2.putText(panel, f"{axes} / {label.replace('_', ' ')}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)
            row.append(panel)
        panels.append(cv2.hconcat(row))
    if not cv2.imwrite(str(path), cv2.vconcat(panels)):
        raise OSError(f"Could not write {path}")


def run(output: Path, depth_checkpoint: Path, device: str, samples: int) -> dict:
    prepared = Path("dataset/prepared_v1")
    phase3_path = PHASE_ROOT / "phase3/visual_hull/phase3_batch_report.json"
    archive = json.loads(phase3_path.read_text(encoding="utf-8"))
    archived = {item["object_id"]: item for item in archive["objects"]}
    resolution = int(archive["resolution"])
    checkpoint = PHASE_ROOT / "checkpoints/whole_jewellery_unet.pt"
    segmenter, weights = _load_model(checkpoint, torch.device(device))
    infer = load_depth_anything_v2(depth_checkpoint, device)
    results = []
    for record in sorted(_manifest(prepared), key=lambda item: (item["split"] != "train", item["object_id"])):
        if record["split"] not in {"train", "val"}:
            continue
        object_id = record["object_id"]
        paths = {view: Path(record["views"][view]["image_path"]) for view in VIEWS}
        images = {view: _read(path) for view, path in paths.items()}
        masks = {view: _predict(segmenter, image, weights, torch.device(device)) for view, image in images.items()}
        base, side, fit, grids = _fit_volume(masks, resolution)
        depth_records = {}
        refined = base
        for view in ("front", "top"):
            depth = np.asarray(infer(paths[view])["depth_m"], dtype=np.float32)
            height, width = masks[view].shape
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
            if not np.isfinite(depth).all() or np.any(depth <= 0):
                raise ValueError(f"Invalid model depth for {object_id}/{view}")
            normalized, _ = _normalized_depth(depth, masks[view])
            small_image = cv2.resize(images[view], (width, height), interpolation=cv2.INTER_AREA)
            agreement = _edge_agreement(small_image, normalized, masks[view])
            depth_records[view] = {"edge_agreement": round(agreement, 6), "eligible": agreement >= EDGE_GATE}
            if agreement >= EDGE_GATE:
                scale = fit["axis_alignment"]["warp_scales_input_px_to_voxel"][view]
                warped, valid = _warp_centered_values(normalized, masks[view], scale, resolution)
                refined, operation = carve_axis_relief(refined, warped, valid, view, STRENGTH, DIRECTION)
                depth_records[view]["operation"] = operation

        removed = int(np.count_nonzero(base) - np.count_nonzero(refined))
        base_mesh = _voxel_mesh(base)
        refined_mesh = _voxel_mesh(refined) if removed else base_mesh
        targets = list((Path("dataset/STL-1") / record["source_object_name"]).glob("*.obj"))
        if len(targets) != 1:
            raise ValueError(f"Expected exactly one paired OBJ for {object_id}")
        metrics, alignment, surfaces = _compare(base_mesh, refined_mesh, _load_mesh(targets[0]), samples)
        folder = output / object_id
        folder.mkdir(parents=True, exist_ok=True)
        preview = folder / "comparison.png"
        _preview(surfaces, preview)
        projections = {
            name: {view: round(_binary_iou(grids[view], mask), 6) for view, mask in _projections(volume).items()}
            for name, volume in (("baseline", base), ("depth_relief", refined))
        }
        result = {
            "object_id": object_id,
            "split": record["split"],
            "prepared_image_sha256": {view: _sha256(path) for view, path in paths.items()},
            "paired_target": {"path": str(targets[0]), "sha256": _sha256(targets[0])},
            "archived_hull_checks": {
                "side_consensus_matches": side == archived[object_id]["selected_side_consensus"],
                "voxel_count_matches": int(np.count_nonzero(base)) == archived[object_id]["volume_voxels"],
                "projection_iou_matches": projections["baseline"] == archived[object_id]["reprojection"]["selected_projection_iou"],
            },
            "depth": depth_records,
            "removed_voxels": removed,
            "surface_metrics": metrics,
            "alignment": alignment,
            "projection_iou": projections,
            "mesh_quality": {
                name: {"watertight": bool(mesh.is_watertight), "body_count": int(mesh.body_count),
                       "euler_number": int(mesh.euler_number)}
                for name, mesh in (("baseline", base_mesh), ("depth_relief", refined_mesh))
            },
            "preview": str(preview),
        }
        _write_json(folder / "report.json", result)
        results.append(result)
        f1 = [item["surface_f_scores"]["1pct"]["f_score"] for item in metrics.values()]
        print(f"{object_id} {record['split']} | removed {removed:,} | F@1% {f1[0]:.4f} → {f1[1]:.4f}", flush=True)

    summaries = {}
    for split in ("train", "val"):
        group = [item for item in results if item["split"] == split]
        summary = {
            "objects": len(group),
            "changed": sum(item["removed_voxels"] > 0 for item in group),
            "mesh_quality_changes": sum(
                item["mesh_quality"]["baseline"] != item["mesh_quality"]["depth_relief"]
                for item in group
            ),
            "all_archived_hull_checks_match": all(
                all(item["archived_hull_checks"].values()) for item in group
            ),
            "scores": {},
        }
        for variant in ("baseline", "depth_relief"):
            scores = [item["surface_metrics"][variant] for item in group]
            summary["scores"][variant] = {
                "mean_chamfer": float(np.mean([item["chamfer_l1_normalized"] for item in scores])),
                **{f"mean_f_score_{key}": float(np.mean([item["surface_f_scores"][key]["f_score"] for item in scores]))
                   for key in ("1pct", "2pct", "5pct")},
            }
        summaries[split] = summary
    report = {
        "schema_version": "image2cad_stl1_depth_relief_v1",
        "method": "front_and_top_depth_anything_v2_bounded_relief",
        "settings": {"resolution": resolution, "edge_gate": EDGE_GATE, "strength": STRENGTH,
                     "direction": DIRECTION, "samples_per_mesh": samples, "device": device,
                     "alignment": "baseline_fit_reused_for_candidate"},
        "baseline_archive_sha256": _sha256(phase3_path),
        "silhouette_checkpoint_sha256": _sha256(checkpoint),
        "depth_checkpoint_sha256": _sha256(depth_checkpoint),
        "claims": {"test_split_evaluated": False, "target_type": "generated_proxy_mesh",
                   "metric_scale_validated": False, "manufacturing_accuracy_validated": False},
        "splits": summaries,
        "objects": results,
    }
    _write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PHASE_ROOT / "stl1_depth_relief_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=50_000)
    args = parser.parse_args()
    if args.samples < 1000:
        parser.error("--samples must be at least 1000")
    print(json.dumps(run(args.output, args.checkpoint, args.device, args.samples)["splits"], indent=2))


if __name__ == "__main__":
    main()
