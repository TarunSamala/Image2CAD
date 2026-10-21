"""Check and render a small, explicitly reviewed training-image observability set.

These labels describe recognizable image content, not correspondence truth,
physical occlusion, geometry or the held-out Phase 5.13 validation split.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from pipeline.evaluate_phase5_training_track_topology import PHASE1_REPORT
from pipeline.extract_vggt_track_evidence import TRAIN_OBJECTS


LABELS = Path("data/phase5_training_observability_review_v1/labels.json")
ROUTES = Path("data/phase5_training_region_routes_v1/report.json")
OUTPUT = Path("data/phase5_training_observability_review_v1")
VIEWS = ("front", "top", "iso")
TILE = (420, 350)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_bbox(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 4 or any(not isinstance(value, (int, float)) for value in values):
        raise ValueError("Expected four numeric normalized bbox coordinates")
    x0, y0, x1, y1 = map(float, values)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"Invalid normalized bbox: {values}")
    return x0, y0, x1, y1


def _panel(image: Image.Image, title: str, *, bbox=None) -> Image.Image:
    width, height = TILE
    panel = Image.new("RGB", TILE, "#f3f3f3")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), title, fill="#171717", font=ImageFont.load_default())
    image = image.convert("RGB")
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        outline = ImageDraw.Draw(image)
        outline.rectangle(
            (round(x0 * image.width), round(y0 * image.height),
             round(x1 * image.width), round(y1 * image.height)),
            outline="#ff2682", width=max(3, image.width // 250),
        )
    contained = ImageOps.contain(image, (width - 18, height - 42))
    panel.paste(contained, ((width - contained.width) // 2,
                            33 + (height - 42 - contained.height) // 2))
    return panel


def _crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    return image.crop((int(x0 * image.width), int(y0 * image.height),
                       int(x1 * image.width), int(y1 * image.height)))


def run(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    labels = json.loads((root / LABELS).read_text(encoding="utf-8"))
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    routes = json.loads((root / ROUTES).read_text(encoding="utf-8"))
    records = {item["object_id"]: item for item in phase1["objects"]}
    route_records = {item["object_id"]: item for item in routes["objects"]}
    rows = []
    seen = set()
    sheet = Image.new("RGB", (4 * TILE[0], len(labels["observations"]) * TILE[1]), "white")
    for row_index, observation in enumerate(labels["observations"]):
        object_id = observation["object_id"]
        if object_id in seen or object_id not in TRAIN_OBJECTS:
            raise ValueError(f"Duplicate or non-training review object: {object_id}")
        seen.add(object_id)
        if set(observation["target_views"]) != {"top", "iso"}:
            raise ValueError(f"Review must explicitly cover top and iso: {object_id}")
        record = records[object_id]
        if record["split"] != "train":
            raise ValueError(f"Not a training object: {object_id}")
        bbox = _checked_bbox(observation["front_bbox_xyxy_fraction"])
        images = {}
        sources = {}
        for view in VIEWS:
            source_record = record["views"][view]
            path = root / source_record["source"]
            actual_hash = _sha256(path)
            if actual_hash != source_record["source_file_sha256"]:
                raise ValueError(f"Source image differs from Phase 1: {path}")
            sources[view] = {"path": path.relative_to(root).as_posix(),
                             "sha256": actual_hash}
            with Image.open(path) as source:
                images[view] = source.convert("RGB")
        target_statuses = {}
        for view in ("top", "iso"):
            status = observation["target_views"][view]["status"]
            if status not in labels["status_vocabulary"]:
                raise ValueError(f"Unknown status: {status}")
            target_statuses[view] = status
        route = route_records[object_id]
        counts = {
            view: sum(cell["target_views"][view]["accepted_track_count"]
                      for cell in route["regions"])
            for view in ("top", "iso")
        }
        panels = (
            _panel(images["front"].copy(), f"{object_id} / FRONT bbox", bbox=bbox),
            _panel(_crop(images["front"], bbox), "FRONT detail (review crop)"),
            _panel(images["top"], f"TOP / {target_statuses['top']}"),
            _panel(images["iso"], f"ISO / {target_statuses['iso']}"),
        )
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * TILE[0], row_index * TILE[1]))
        rows.append({
            "object_id": object_id,
            "split": "train",
            "region_id": observation["region_id"],
            "front_bbox_xyxy_fraction": list(bbox),
            "sources": sources,
            "target_views": deepcopy(observation["target_views"]),
            "machine_accepted_tracks_whole_front": counts,
            "track_count_scope": "whole front relief query set, not the annotated face bbox",
        })
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "training_observability_review.png"
    sheet.save(preview, optimize=True)
    report = {
        "schema_version": "lalitha_phase5_training_observability_review_v1",
        "stage": "Phase 5.21 small training-image content review",
        "label_source": LABELS.as_posix(),
        "scope": {"split": "train", "object_ids": sorted(seen),
                  "source_view": "front", "reviewed_target_views": ["top", "iso"],
                  "unreviewed_views": labels["unreviewed_views"]},
        "status_vocabulary": labels["status_vocabulary"],
        "observations": rows,
        "summary": {
            "object_count": len(rows),
            "top_without_recognizable_face_detail": sum(
                row["target_views"]["top"]["status"] == "no_recognizable_detail"
                for row in rows
            ),
            "iso_with_partial_recognizable_face_detail": sum(
                row["target_views"]["iso"]["status"] == "recognizable_partial"
                for row in rows
            ),
            "top_tracks_can_exist_without_reviewed_face_detail": any(
                row["machine_accepted_tracks_whole_front"]["top"] > 0
                and row["target_views"]["top"]["status"] == "no_recognizable_detail"
                for row in rows
            ),
        },
        "checks": {"all_training": all(row["split"] == "train" for row in rows),
                   "source_files_match_phase1": True,
                   "view_statuses_explicit": all(set(row["target_views"]) == {"top", "iso"}
                                                 for row in rows),
                   "validation_test_untouched": True},
        "claims": {"point_correspondence_ground_truth": False,
                   "physical_occlusion_proven": False,
                   "full_training_set_reviewed": False,
                   "matching_accuracy_improved": False,
                   "surface_promotion_authorized": False,
                   "cad_fusion_authorized": False},
        "decision": "expand_review_to_more_training_regions_before_learning_or_gating_routes",
        "limitations": labels["limitations"],
        "artifacts": {"preview": preview.relative_to(root).as_posix()
                      if preview.is_relative_to(root) else preview.as_posix()},
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
