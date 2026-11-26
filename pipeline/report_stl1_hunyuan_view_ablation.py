"""Summarize the controlled one-view versus three-view Hunyuan pilot."""

from __future__ import annotations

import json
from statistics import mean

import cv2
import numpy as np

from pipeline.pilot_stl1_hunyuan import OBJECTS, OUTPUT
from pipeline.run_saraswati_blind_benchmark import _sha256, _write_json


FIXED_SETTINGS = (
    "model_revision", "source_revision", "config_sha256", "checkpoint_sha256",
    "input_manifest_sha256", "seed", "steps", "octree_resolution", "device",
)


def _scores(report, generation):
    metrics = report["metrics"]["hunyuan"]
    openings = metrics["angled"]["enclosed_openings_largest_first"]
    return {
        "angled_iou": metrics["angled"]["silhouette_iou"],
        "head_side_iou": metrics["head_side_iou"],
        "shank_side_iou": metrics["shank_side_iou"],
        "largest_opening_empty_fraction": openings[0]["projected_empty_fraction"] if openings else None,
        "paired_obj_f_score_at_1pct": metrics["paired_obj_agreement"]["surface_f_scores"]["1pct"]["f_score"],
        "watertight": metrics["watertight"],
        "connected_bodies": metrics["connected_bodies"],
        "faces": metrics["faces"],
        "inference_seconds": generation["inference_seconds"],
    }


def run():
    rows = []
    sheets = []
    for object_id in OBJECTS:
        original = OUTPUT / object_id
        variants = {"front_only": original / "front-only", "three_view": original}
        runs = {name: json.loads((folder / "generation.json").read_text(encoding="utf-8"))
                for name, folder in variants.items()}
        reports = {name: json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
                   for name, folder in variants.items()}
        single, multi = runs["front_only"], runs["three_view"]
        if any(single[key] != multi[key] for key in FIXED_SETTINGS):
            raise ValueError(f"Nonmatching generation settings on {object_id}")
        if single["input_views"] != ["front"] or multi["input_views"] != ["front", "left", "right"]:
            raise ValueError(f"Wrong input views on {object_id}")
        if (reports["front_only"]["evaluation_camera"] != reports["three_view"]["evaluation_camera"]
                or reports["front_only"]["angled_reference_sha256"] != reports["three_view"]["angled_reference_sha256"]
                or reports["front_only"]["alignment_reference"] != reports["three_view"]["alignment_reference"]
                or reports["front_only"]["metrics"]["visual_hull"] != reports["three_view"]["metrics"]["visual_hull"]):
            raise ValueError(f"Evaluation reference changed on {object_id}")
        for name in variants:
            if reports[name]["candidate_sha256"] != runs[name]["mesh_sha256"]:
                raise ValueError(f"Comparison is out of date on {object_id}/{name}")
        manifest_path = original / "input/manifest.json"
        if _sha256(manifest_path) != single["input_manifest_sha256"]:
            raise ValueError(f"Input manifest changed on {object_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = {
            "object_id": object_id, "split": reports["three_view"]["split"],
            "fixed_settings": {key: single[key] for key in FIXED_SETTINGS},
            "front_image_sha256": manifest["inputs"]["front"]["sha256"],
            "angled_reference_sha256": reports["three_view"]["angled_reference_sha256"],
            "front_only_trace": single["conditioning_trace"],
            "scores": {name: _scores(reports[name], runs[name]) for name in variants},
            "artifacts": {name: str(folder / "comparison.json") for name, folder in variants.items()},
        }
        rows.append(row)
        # Existing preview columns contain the same angled photo and camera render.
        cells = []
        for name, folder in variants.items():
            image = cv2.imread(str(folder / "shape_comparison.png"))
            if image is None or image.shape[0] % 2 or image.shape[1] % 3:
                raise ValueError(f"Cannot read comparison preview in {folder}")
            half_height, third_width = image.shape[0] // 2, image.shape[1] // 3
            if not cells:
                cells.append(image[half_height:, :third_width].copy())
            shaded = image[half_height:, third_width:2 * third_width].copy()
            shaded[:28] = 248
            score = row["scores"][name]["angled_iou"]
            cv2.putText(shaded, f"{name.replace('_', ' ').upper()}  IoU {score:.3f}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
            cells.append(shaded)
        cells[0][:28] = 248
        cv2.putText(cells[0], f"{object_id.upper()} / UNUSED ANGLED", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        sheets.append(np.hstack(cells))
    preview = OUTPUT / "view_ablation.png"
    if not cv2.imwrite(str(preview), np.vstack(sheets)):
        raise OSError(f"Could not write {preview}")
    summary = {
        variant: {
            key: round(mean(row["scores"][variant][key] for row in rows), 6)
            for key in ("angled_iou", "head_side_iou", "shank_side_iou", "paired_obj_f_score_at_1pct")
        }
        for variant in ("front_only", "three_view")
    }
    _write_json(OUTPUT / "view_ablation.json", {
        "experiment": "Same Hunyuan 2mv Turbo model with front only versus front, left and right",
        "limits": [
            "Three selected STL-1 rings and a single fixed seed.",
            "Each candidate was aligned to the four-view hull before the angled comparison.",
            "The head and shank measurements are image zones. The paired OBJs are generated proxies.",
        ],
        "summary": summary, "objects": rows, "preview": str(preview),
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
