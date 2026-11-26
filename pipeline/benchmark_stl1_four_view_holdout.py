"""Build Studio's four-view hulls and audit them against unused angled images."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.reconstruct_dataset_phase3 import _cell, _error_overlay, _manifest, _read
from pipeline.refine_phase3_1 import _projection
from pipeline.run_saraswati_blind_benchmark import (
    _foreground_mask,
    _load_mesh,
    _sha256,
    _write_json,
    reconstruct_from_images,
)
from pipeline.train_dataset_phases import _binary_iou, _boundary_f1
from pipeline.validate_phase3_2 import _basis_camera


VIEWS = ("front", "top", "lsv", "rsv")
PHASE_ROOT = Path("dataset/phase_runs/v1")
OUTPUT = PHASE_ROOT / "stl1_four_view_holdout_v1"
ARCHIVE = PHASE_ROOT / "phase3/visual_hull/phase3_batch_report.json"
RESOLUTION = 128
AUDIT_SIZE = 256
MARGIN = 12


def _normalize_mask(mask):
    """Center the target in a fixed frame and preserve its aspect ratio."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if not w or not h:
        raise ValueError("Empty evaluation silhouette")
    scale = (AUDIT_SIZE - 2 * MARGIN - 1) / max(w - 1, h - 1, 1)
    center = (AUDIT_SIZE - 1) / 2
    transform = np.array([
        [scale, 0, center - scale * (x + (w - 1) / 2)],
        [0, scale, center - scale * (y + (h - 1) / 2)],
    ])
    return cv2.warpAffine(
        mask.astype(np.uint8), transform, (AUDIT_SIZE, AUDIT_SIZE), flags=cv2.INTER_NEAREST
    ) > 0


def _angle_candidates():
    # Hull Z increases downwards. Evaluation flips Y and Z with a proper rotation.
    for azimuth in (-60, -45, -30, 30, 45, 60, -150, -135, -120, 120, 135, 150):
        for elevation in (20, 40):
            a, e = np.deg2rad([azimuth, elevation])
            direction = (np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e))
            yield azimuth, elevation, _basis_camera("stl1_iso", direction)


def _project(mesh, camera):
    frame = (MARGIN, MARGIN, AUDIT_SIZE - MARGIN - 1, AUDIT_SIZE - MARGIN - 1)
    x, y, _, _ = _projection(mesh, camera, frame)
    polygons = np.rint(np.column_stack([x, y])[mesh.faces]).astype(np.int32)
    # Union projected triangles. A single fillPoly call can apply even/odd parity
    # where front and back faces overlap.
    result = np.zeros((AUDIT_SIZE, AUDIT_SIZE), np.uint8)
    for triangle in polygons:
        cv2.fillConvexPoly(result, triangle, 1)
    return result > 0


def _opening_metrics(target, projection):
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~target).astype(np.uint8))
    outside = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    enclosed = [idx for idx in range(1, count) if idx not in outside
                and stats[idx, cv2.CC_STAT_AREA] >= 16]
    enclosed.sort(key=lambda idx: stats[idx, cv2.CC_STAT_AREA], reverse=True)
    return [
        {"target_area_px": int(stats[idx, cv2.CC_STAT_AREA]),
         "projected_empty_fraction": round(float(np.mean(~projection[labels == idx])), 6)}
        for idx in enclosed
    ]


def _score(target, projection):
    return {
        "silhouette_iou": round(_binary_iou(target, projection), 6),
        "boundary_f1_2px": round(_boundary_f1(target, projection), 6),
        "enclosed_openings_largest_first": _opening_metrics(target, projection),
    }


def _load_upright(path):
    mesh = _load_mesh(path)
    mesh.vertices = np.asarray(mesh.vertices) * np.array([1, -1, -1])
    return mesh


def _summary(rows):
    return {
        "rings": len(rows),
        **{
            variant: {
                "mean_iso_iou": round(float(np.mean([row[variant]["silhouette_iou"] for row in rows])), 6),
                "min_iso_iou": min(row[variant]["silhouette_iou"] for row in rows),
                "mean_iso_boundary_f1": round(float(np.mean([row[variant]["boundary_f1_2px"] for row in rows])), 6),
            }
            for variant in ("four_view", "archived_hull")
        },
    }


def run():
    records = _manifest(Path("dataset/prepared_v1"))
    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    archived = {row["object_id"]: row for row in archive["objects"]}
    angles = list(_angle_candidates())
    sums = np.zeros(len(angles), dtype=np.float64)
    rows = []
    cohort = []
    for split in ("train", "val"):
        if split == "val":
            selected_index = int(np.argmax(sums))
            azimuth, elevation, camera = angles[selected_index]
            print(f"Frozen angled camera: azimuth {azimuth}°, elevation {elevation}°", flush=True)
        for record in (item for item in records if item["split"] == split):
            object_id = record["object_id"]
            folder = OUTPUT / object_id
            sources = {view: Path(record["views"][view]["image_path"]) for view in VIEWS}
            mesh_path, record_path = reconstruct_from_images(
                sources, folder, RESOLUTION, object_id=object_id, dataset_name="STL-1",
                proposal_filename="four_view_proposal.obj",
            )
            if split == "train":
                mesh = _load_upright(mesh_path)
                iso_path = Path(record["views"]["iso"]["image_path"])
                target = _normalize_mask(_foreground_mask(iso_path)[0])
                sums += [_binary_iou(target, _project(mesh, item[2])) for item in angles]
            cohort.append((record, mesh_path, record_path))
            print(f"Built {object_id} ({split})", flush=True)

    for record, mesh_path, record_path in cohort:
        object_id = record["object_id"]
        prior = archived[object_id]
        iso_path = Path(record["views"]["iso"]["image_path"])
        target = _normalize_mask(_foreground_mask(iso_path)[0])
        mesh = _load_upright(mesh_path)
        old = _load_upright(Path(prior["exports"]["stl"]))
        current_projection, old_projection = _project(mesh, camera), _project(old, camera)
        reconstruction = json.loads(record_path.read_text(encoding="utf-8"))
        comparison = np.hstack([
            _cell(_read(iso_path), f"{object_id.upper()} UNUSED ISO"),
            _cell(target, "REFERENCE SILHOUETTE"),
            _cell(_error_overlay(target, current_projection), "NEW FOUR-VIEW HULL"),
            _cell(_error_overlay(target, old_projection), "ARCHIVED HULL"),
        ])
        preview = mesh_path.parent / "angled_comparison.png"
        if not cv2.imwrite(str(preview), comparison):
            raise OSError(f"Could not write {preview}")
        row = {
            "object_id": object_id, "split": record["split"],
            "iso_path": str(iso_path), "iso_sha256": _sha256(iso_path),
            "generation_record": str(record_path), "mesh": str(mesh_path),
            "preview": str(preview),
            "input_view_iou": reconstruction["fit"]["selected_projection_iou"],
            "watertight": bool(mesh.is_watertight), "bodies": int(mesh.body_count),
            "four_view": _score(target, current_projection),
            "archived_hull": _score(target, old_projection),
        }
        _write_json(mesh_path.parent / "evaluation.json", row)
        rows.append(row)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    report = {
        "schema_version": "image2cad_stl1_four_view_holdout_v1",
        "resolution": RESOLUTION,
        "generation_input_views": list(VIEWS),
        "audit_view": "iso",
        "camera": {
            "azimuth": azimuth, "elevation": elevation,
            "selection": "highest mean normalized silhouette IoU on train only; fixed for validation and both hulls",
            "candidates": [{"azimuth": a, "elevation": e, "train_mean_iou": round(float(total / len(train)), 6)}
                           for (a, e, _), total in zip(angles, sums)],
            "in_plane_alignment": "independent centering and isotropic bounding-box extent normalization",
        },
        "archive_sha256": _sha256(ARCHIVE),
        "claims": {"iso_used_for_mesh_generation": False, "test_split_accessed": False,
                   "calibrated_pose": False, "physical_scale_validated": False},
        "splits": {"train_camera_selection": _summary(train), "validation": _summary(val)},
        "objects": rows,
    }
    _write_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run()["splits"], indent=2))
