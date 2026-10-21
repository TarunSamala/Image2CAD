"""Audit frozen VGGT tracks inside the three reviewed front-face boxes.

The reviewed boxes are visual regions, not masks or point-correspondence truth.
The Phase 5.5 reciprocal gate is reused without tuning on validation/test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    evaluate_track_consistency,
)
from pipeline.evaluate_phase5_training_track_topology import (
    BASELINE_TRACK_ROOT,
    HYBRID_TRACK_ROOT,
    PHASE1_REPORT,
    SOURCE_INDEX,
    _load_tracks,
)
from pipeline.extract_vggt_track_evidence import IMAGE_SIZE, TRAIN_OBJECTS, VIEW_ORDER, _square


LABELS = Path("data/phase5_training_observability_review_v1/labels.json")
MANIFEST = Path("dataset/prepared_v1/manifest.jsonl")
OUTPUT = Path("data/phase5_reviewed_face_tracks_v1")


def _review_bbox_in_track_pixels(
    fractions: list[float], view: dict, size: int = IMAGE_SIZE
) -> tuple[float, float, float, float]:
    """Map an original-source fractional bbox through the frozen prepared transform."""
    if len(fractions) != 4 or not (0 <= fractions[0] < fractions[2] <= 1
                                   and 0 <= fractions[1] < fractions[3] <= 1):
        raise ValueError("Invalid original-source bbox fractions")
    width, height = view["source_size_wh"]
    crop_x0, crop_y0, _, _ = view["transform"]["source_crop_xyxy"]
    scale = view["transform"]["resize_scale"]
    place_x, place_y, _, _ = view["transform"]["placement_xywh"]
    prepared_width, prepared_height = view["prepared_size_wh"]
    x0 = (place_x + (fractions[0] * width - crop_x0) * scale) * size / prepared_width
    x1 = (place_x + (fractions[2] * width - crop_x0) * scale) * size / prepared_width
    y0 = (place_y + (fractions[1] * height - crop_y0) * scale) * size / prepared_height
    y1 = (place_y + (fractions[3] * height - crop_y0) * scale) * size / prepared_height
    if not (0 <= x0 < x1 <= size and 0 <= y0 < y1 <= size):
        raise ValueError("Reviewed bbox falls outside prepared track image")
    return tuple(round(float(value), 4) for value in (x0, y0, x1, y1))


def _render(rows: list[dict], root: Path, output: Path) -> None:
    figure, axes = plt.subplots(len(rows), 3, figsize=(11.5, 4.0 * len(rows)),
                                 facecolor="#f5f5f5", squeeze=False)
    for row_index, row in enumerate(rows):
        object_id = row["object_id"]
        record = row["prepared_views"]
        source = row["source_query_points_xy"]
        selected = row["source_inside_review_bbox"]
        x0, y0, x1, y1 = row["review_bbox_xyxy_track_pixels"]
        for column, view in enumerate(("front", "top", "iso")):
            axis = axes[row_index, column]
            image = _square(root / record[view], IMAGE_SIZE)
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_xlim(0, IMAGE_SIZE)
            axis.set_ylim(IMAGE_SIZE, 0)
            axis.axis("off")
            if view == "front":
                axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         fill=False, edgecolor="#ff2785", linewidth=2))
                points = np.asarray(source, np.float32)[np.asarray(selected, bool)]
                axis.scatter(points[:, 0], points[:, 1], s=9, c="#fff24f", alpha=0.8)
                axis.set_title(f"{object_id} FRONT: {row['review_bbox_query_count']} queries")
            else:
                target = row["target_views"][view]
                points = np.asarray(target["accepted_target_points_xy"], np.float32).reshape(-1, 2)
                if len(points):
                    axis.scatter(points[:, 0], points[:, 1], s=16, c="#ff2785",
                                 marker="o", edgecolors="white", linewidths=0.4)
                axis.set_title(f"{view.upper()}: {target['accepted_inside_review_bbox']} accepted from box")
    figure.suptitle("PHASE 5.22 - REVIEWED FACE REGION / FROZEN TRACK AUDIT", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    labels = json.loads((root / LABELS).read_text(encoding="utf-8"))
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_records = {item["object_id"]: item for item in phase1["objects"]}
    prepared_records = {
        item["object_id"]: item
        for item in (json.loads(line) for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines())
    }
    metric_depths = _metric_depth_records(root)
    direct_normals = _metric3d_normal_records(root)
    rows = []
    for observation in labels["observations"]:
        object_id = observation["object_id"]
        if object_id not in TRAIN_OBJECTS or phase1_records[object_id]["split"] != "train":
            raise ValueError(f"Reviewed object is not in training split: {object_id}")
        prepared = prepared_records[object_id]
        front = prepared["views"]["front"]
        if front["source_sha256"] != phase1_records[object_id]["views"]["front"]["source_file_sha256"]:
            raise ValueError(f"Prepared/Phase 1 front source mismatch: {object_id}")
        prepared_sha256 = hashlib.sha256((root / front["image_path"]).read_bytes()).hexdigest()
        if prepared_sha256 != phase1_records[object_id]["views"]["front"]["normalized_source_file_sha256"]:
            raise ValueError(f"Prepared front image differs from Phase 1: {object_id}")
        box = _review_bbox_in_track_pixels(observation["front_bbox_xyxy_fraction"], front)
        baseline = _load_tracks(root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        tracks = _load_tracks(root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz")
        if not np.allclose(tracks["query_points_xy"][:, :384], baseline["query_points_xy"]):
            raise ValueError(f"Baseline queries changed: {object_id}")
        evidence = _load_view_evidence(root, phase1_records[object_id], metric_depths, direct_normals)
        evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(tracks, evidence),
            evaluate_track_consistency(baseline, evidence),
        )
        points = tracks["query_points_xy"][SOURCE_INDEX]
        x0, y0, x1, y1 = box
        inside = ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                  & (points[:, 1] >= y0) & (points[:, 1] <= y1))
        candidate = evaluated["candidate"][SOURCE_INDEX]
        targets = {}
        for view in ("top", "iso"):
            target_index = VIEW_ORDER.index(view)
            accepted = evaluated["accepted"][SOURCE_INDEX, target_index] & inside & candidate
            target_points = tracks["tracks_xy"][SOURCE_INDEX, target_index, accepted]
            targets[view] = {
                "visual_review_status": observation["target_views"][view]["status"],
                "accepted_inside_review_bbox": int(np.count_nonzero(accepted)),
                "accepted_whole_front": int(np.count_nonzero(
                    evaluated["accepted"][SOURCE_INDEX, target_index] & candidate)),
                "accepted_target_points_xy": np.round(target_points, 2).tolist(),
            }
        rows.append({
            "object_id": object_id,
            "split": "train",
            "review_bbox_xyxy_track_pixels": box,
            "review_bbox_query_count": int(np.count_nonzero(inside)),
            "review_bbox_relief_candidate_count": int(np.count_nonzero(inside & candidate)),
            "target_views": targets,
            "prepared_views": {view: prepared["views"][view]["image_path"]
                               for view in ("front", "top", "iso")},
            "prepared_front_sha256": prepared_sha256,
            "source_query_points_xy": np.round(points, 2).tolist(),
            "source_inside_review_bbox": inside.tolist(),
            "checks": {
                "top_subset_of_whole_front": targets["top"]["accepted_inside_review_bbox"]
                <= targets["top"]["accepted_whole_front"],
                "iso_subset_of_whole_front": targets["iso"]["accepted_inside_review_bbox"]
                <= targets["iso"]["accepted_whole_front"],
            },
        })
    output.mkdir(parents=True, exist_ok=True)
    audit = output / "reviewed_face_track_audit.png"
    _render(rows, root, audit)
    report_rows = [
        {key: value for key, value in row.items()
         if key not in {"source_query_points_xy", "source_inside_review_bbox"}}
        for row in rows
    ]
    report = {
        "schema_version": "lalitha_phase5_reviewed_face_tracks_v1",
        "stage": "Phase 5.22 training-only reviewed region track audit",
        "scope": {"split": "train", "object_ids": [row["object_id"] for row in rows],
                  "source_view": "front", "target_views": ["top", "iso"]},
        "source_labels": LABELS.as_posix(),
        "mapping": "original source fractional bbox -> prepared crop/scale/placement -> 518px VGGT query image",
        "frozen_reciprocal_gate_reused": True,
        "objects": report_rows,
        "summary": {
            "object_count": len(rows),
            "face_bbox_candidate_count": sum(row["review_bbox_relief_candidate_count"] for row in rows),
            "top_accepted_inside_review_boxes": sum(row["target_views"]["top"]["accepted_inside_review_bbox"] for row in rows),
            "iso_accepted_inside_review_boxes": sum(row["target_views"]["iso"]["accepted_inside_review_bbox"] for row in rows),
            "top_accepted_whole_front": sum(row["target_views"]["top"]["accepted_whole_front"] for row in rows),
        },
        "checks": {"training_only": all(row["split"] == "train" for row in rows),
                   "prepared_front_matches_phase1": True,
                   "subset_counts_valid": all(all(row["checks"].values()) for row in rows),
                   "validation_and_test_untouched": True,
                   "phase5_13_gate_unchanged": True},
        "claims": {"face_correspondence_validated": False,
                   "physical_occlusion_proven": False,
                   "matching_accuracy_improved": False,
                   "surface_promotion_authorized": False,
                   "cad_fusion_authorized": False},
        "limitations": [
            "Three selected training face boxes do not estimate population accuracy.",
            "Approximate boxes are not pixel-accurate face segmentation.",
            "Machine-accepted points are not ground-truth 3D surface identities.",
            "Visual absence in a target image is not proof of a hidden physical surface.",
        ],
        "artifacts": {"audit": audit.relative_to(root).as_posix()},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
