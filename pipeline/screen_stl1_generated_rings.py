"""Screen pilot meshes against the top image and basic structure checks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from trimesh.sample import sample_surface

from pipeline.benchmark_stl1_four_view_holdout import (
    OUTPUT as HULL_ROOT, _load_upright, _normalize_mask, _opening_metrics,
)
from pipeline.pilot_stl1_hunyuan import OBJECTS, OUTPUT as HUNYUAN_ROOT
from pipeline.pilot_stl1_hunyuan_relief_transfer import FRAME, _project_fixed
from pipeline.pilot_stl1_triposr import OUTPUT as TRIPOSR_ROOT
from pipeline.reconstruct_dataset_phase3 import _cell, _error_overlay, _manifest, _read
from pipeline.refine_phase3_1 import _jewelry_transform
from pipeline.run_saraswati_blind_benchmark import _foreground_mask, _load_mesh, _normalize, _sha256, _write_json
from pipeline.train_dataset_phases import _binary_iou
from pipeline.validate_phase3_2 import _basis_camera


OUTPUT = Path("dataset/phase_runs/v1/stl1_generator_structure_screen_v1")
MINIMUM_OPEN_FRACTION = 0.90
MAXIMUM_OPEN_FRACTION_LOSS = 0.03
MINIMUM_OPENING_AREA = 0.01
MINIMUM_TOP_IOU = 0.75
MAXIMUM_TOP_IOU_LOSS = 0.10


def _candidate(folder, generator, base, base_path):
    path = folder / f"{generator}_shape.obj"
    report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    digest = _sha256(path)
    if digest != report["candidate_sha256"] or str(base_path) != report["alignment_reference"]:
        raise ValueError(f"Stale candidate alignment: {path}")
    _, center, extent = _normalize(sample_surface(base, 12000, seed=456)[0])
    transform = report[f"{generator}_alignment"]
    mesh = _load_mesh(path)
    normalized = (np.asarray(mesh.vertices) - transform["source_center"]) / transform["source_extent"]
    mesh.vertices = (normalized @ np.asarray(transform["rotation_source_to_target"]).T
                     + transform["normalized_translation"]) * extent + center
    return mesh, digest


def _screen(mesh, target, projection, minimum_open_fraction, minimum_top_iou):
    openings = _opening_metrics(target, projection)
    largest = openings[0] if openings and openings[0]["target_area_px"] >= target.size * MINIMUM_OPENING_AREA else None
    iou = _binary_iou(target, projection)
    checks = {
        "watertight": bool(mesh.is_watertight),
        "consistent_winding": bool(mesh.is_winding_consistent),
        "single_connected_body": mesh.body_count == 1,
        "finger_opening_visible_and_preserved": largest is not None
        and largest["projected_empty_fraction"] >= minimum_open_fraction,
        "top_silhouette_preserved": iou >= minimum_top_iou,
    }
    return {
        "eligible_for_next_geometry_review": all(checks.values()),
        "checks": checks,
        "connected_bodies": int(mesh.body_count),
        "top_silhouette_iou": round(iou, 6),
        "largest_top_opening": largest,
    }


def run():
    records = {row["object_id"]: row for row in _manifest(Path("dataset/prepared_v1"))}
    hull = json.loads((HULL_ROOT / "report.json").read_text(encoding="utf-8"))
    baselines = {row["object_id"]: row for row in hull["objects"]}
    camera = _basis_camera("top", (0, 0, 1))
    rows, panels = [], []
    for object_id in OBJECTS:
        row = baselines[object_id]
        base_path = Path(row["mesh"])
        generation = json.loads(Path(row["generation_record"]).read_text(encoding="utf-8"))
        top_path = Path(records[object_id]["views"]["top"]["image_path"])
        top_hash = _sha256(top_path)
        base_hash = _sha256(base_path)
        if (top_hash != generation["source_images"]["top"]["sha256"]
                or base_hash != generation["proposal"]["sha256"]):
            raise ValueError(f"Baseline inputs changed for {object_id}")
        base = _load_upright(base_path)
        target = _normalize_mask(_foreground_mask(top_path)[0])
        transform = _jewelry_transform(base, camera, FRAME)
        baseline_projection = _project_fixed(base, camera, transform)
        baseline_screen = _screen(base, target, baseline_projection, MINIMUM_OPEN_FRACTION, MINIMUM_TOP_IOU)
        opening = baseline_screen["largest_top_opening"]
        minimum = max(MINIMUM_OPEN_FRACTION, opening["projected_empty_fraction"] - MAXIMUM_OPEN_FRACTION_LOSS) if opening else MINIMUM_OPEN_FRACTION
        minimum_top_iou = max(MINIMUM_TOP_IOU, baseline_screen["top_silhouette_iou"] - MAXIMUM_TOP_IOU_LOSS)
        score = {"visual_hull": baseline_screen}
        previews = [
            _cell(_read(top_path), f"{object_id.upper()} TOP SOURCE"),
            _cell(_error_overlay(target, baseline_projection), "FOUR-VIEW HULL"),
        ]
        for label, folder, generator in (
            ("hunyuan_front", HUNYUAN_ROOT / object_id / "front-only", "hunyuan"),
            ("hunyuan_three", HUNYUAN_ROOT / object_id, "hunyuan"),
            ("triposr", TRIPOSR_ROOT / object_id, "triposr"),
        ):
            if not (folder / "comparison.json").is_file():
                score[label] = {"status": "not_available"}
                previews.append(_cell(np.full(target.shape, 248, dtype=np.uint8), f"{label.upper()} PENDING"))
                continue
            mesh, digest = _candidate(folder, generator, base, base_path)
            projection = _project_fixed(mesh, camera, transform)
            score[label] = {**_screen(mesh, target, projection, minimum, minimum_top_iou), "mesh_sha256": digest}
            opening = score[label]["largest_top_opening"]
            fraction = "unresolved" if opening is None else f'{opening["projected_empty_fraction"]:.0%} open'
            previews.append(_cell(_error_overlay(target, projection), f"{label.upper()} / {fraction}"))
        panels.append(np.hstack(previews))
        rows.append({"object_id": object_id, "split": row["split"], "top_image_sha256": top_hash,
                     "baseline_sha256": base_hash, "minimum_open_fraction": round(minimum, 6),
                     "minimum_top_iou": round(minimum_top_iou, 6), "scores": score})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preview = OUTPUT / "top_opening_comparison.png"
    if not cv2.imwrite(str(preview), np.vstack(panels)):
        raise OSError(f"Could not save {preview}")
    _write_json(OUTPUT / "report.json", {
        "method": "common top camera and hull framing; largest enclosed background region from the top input view",
        "decision_scope": "provisional geometry screen for these sculptural rings",
        "limits": ["Axis camera and hull alignment are hypotheses.",
                   "This does not check self-intersections, physical dimensions or manufacturing suitability.",
                   "A largest enclosed silhouette region is a proxy for the finger opening."],
        "settings": {"minimum_open_fraction": MINIMUM_OPEN_FRACTION,
                     "maximum_loss_relative_to_hull": MAXIMUM_OPEN_FRACTION_LOSS,
                     "minimum_top_iou": MINIMUM_TOP_IOU, "maximum_top_iou_loss_relative_to_hull": MAXIMUM_TOP_IOU_LOSS,
                     "minimum_opening_fraction_of_image": MINIMUM_OPENING_AREA},
        "objects": rows, "preview": str(preview),
    })


if __name__ == "__main__":
    run()
