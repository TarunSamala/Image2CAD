"""Compare local and whole-front routing signals on reviewed training regions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.audit_phase5_reviewed_face_tracks import (
    MANIFEST,
    _review_bbox_in_track_pixels,
)
from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.build_phase5_training_observability_review import TILE, _crop, _panel, _sha256
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
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS, VIEW_ORDER


OUTPUT = Path("data/phase5_training_region_review_v1")
TARGET_VIEWS = ("top", "iso")


def run(root: Path) -> dict:
    output = root / OUTPUT
    labels_path = output / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["regions"]
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    records = {item["object_id"]: item for item in phase1["objects"]}
    prepared = {
        record["object_id"]: record
        for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }
    metric_depths = _metric_depth_records(root)
    normals = _metric3d_normal_records(root)
    sheet = Image.new("RGB", (4 * TILE[0], len(labels) * TILE[1]), "white")
    rows = []
    for object_id in sorted({label["object_id"] for label in labels}):
        record = records[object_id]
        if object_id not in TRAIN_OBJECTS or record["split"] != "train" or prepared[object_id]["split"] != "train":
            raise ValueError(f"Reviewed object is outside the training split: {object_id}")
        views = prepared[object_id]["views"]
        images = {}
        hashes = {}
        for view in ("front", *TARGET_VIEWS):
            source = record["views"][view]
            path = root / source["source"]
            hashes[view] = _sha256(path)
            if hashes[view] != source["source_file_sha256"] or hashes[view] != views[view]["source_sha256"]:
                raise ValueError(f"Review image differs from frozen source: {path}")
            with Image.open(path) as image:
                images[view] = image.convert("RGB")
        for view in VIEW_ORDER:
            frozen_hash = record["views"][view]["normalized_source_file_sha256"]
            if _sha256(root / views[view]["image_path"]) != frozen_hash:
                raise ValueError(f"Prepared track image differs from Phase 1: {object_id}/{view}")
        baseline_path = root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        hybrid_path = root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        baseline = _load_tracks(baseline_path)
        hybrid = _load_tracks(hybrid_path)
        if not np.allclose(hybrid["query_points_xy"][:, :384], baseline["query_points_xy"]):
            raise ValueError(f"Baseline queries changed: {object_id}")
        evidence = _load_view_evidence(root, record, metric_depths, normals)
        tracks = _preserve_baseline_gate(
            evaluate_track_consistency(hybrid, evidence),
            evaluate_track_consistency(baseline, evidence),
        )
        candidates = tracks["candidate"][SOURCE_INDEX]
        x, y = hybrid["query_points_xy"][SOURCE_INDEX].T
        for label in (label for label in labels if label["object_id"] == object_id):
            box = label["front_bbox"]
            x0, y0, x1, y1 = _review_bbox_in_track_pixels(box, views["front"])
            local = candidates & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            targets = {}
            for view in TARGET_VIEWS:
                if label[view] not in {"absent", "partial", "uncertain"}:
                    raise ValueError(f"Unknown review status for {object_id}/{view}")
                accepted = tracks["accepted"][SOURCE_INDEX, VIEW_ORDER.index(view)]
                targets[view] = {
                    "review_status": label[view],
                    "local_accepted_tracks": int(np.count_nonzero(accepted & local)),
                    "whole_front_accepted_tracks": int(np.count_nonzero(accepted & candidates)),
                }
            panels = (
                _panel(images["front"].copy(), f"{object_id} / FRONT", bbox=box),
                _panel(_crop(images["front"], box), label["region"]),
                *(_panel(images[view], f"{view.upper()} / {label[view]}") for view in TARGET_VIEWS),
            )
            for column, panel in enumerate(panels):
                sheet.paste(panel, (column * TILE[0], len(rows) * TILE[1]))
            rows.append({
                "object_id": object_id,
                "region": label["region"],
                "front_bbox_fraction": box,
                "front_bbox_track_pixels": [x0, y0, x1, y1],
                "local_relief_candidates": int(np.count_nonzero(local)),
                "note": label["note"],
                "source_image_hashes": hashes,
                "targets": targets,
            })
    preview = output / "training_region_review.png"
    sheet.save(preview, optimize=True)
    summary = {}
    for view in TARGET_VIEWS:
        counts = Counter()
        for row in rows:
            region = row["targets"][view]
            label = region["review_status"]
            counts[f"{label}_regions"] += 1
            if label != "uncertain":
                for scope in ("local", "whole_front"):
                    counts[f"{label}_with_{scope}_support"] += int(region[f"{scope}_accepted_tracks"] > 0)
        summary[view] = dict(counts)
    report = {
        "stage": "Phase 5.23 training region routing review",
        "split": "train",
        "routing_probe": "Any accepted reciprocal track in the region versus anywhere in the front view.",
        "scope": "These selected training examples measure agreement with region visibility labels. "
                 "A recognizable component can still lack the same visible surface in both images.",
        "labels_sha256": _sha256(labels_path),
        "region_count": len(rows),
        "summary": summary,
        "regions": rows,
        "preview": preview.relative_to(root).as_posix(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd())["summary"], indent=2))
