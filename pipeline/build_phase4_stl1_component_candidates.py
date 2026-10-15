"""Inventory image-supported component candidates across the STL-1 dataset.

The STL-1 renders are monochrome and have no component labels.  This stage
therefore extracts observable enclosed negative spaces, but deliberately does
not relabel circular openings or relief ornaments as gemstones.  Its output is
review evidence for Phase 4, not component ground truth or CAD authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("front", "top", "iso", "lsv", "rsv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, mode: int) -> np.ndarray:
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _enclosed_components(mask: np.ndarray) -> list[tuple[np.ndarray, dict[str, Any]]]:
    foreground = mask > 0
    background = (~foreground).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        background, connectivity=8
    )
    foreground_area = max(1, int(np.count_nonzero(foreground)))
    minimum_area = max(48, int(round(foreground_area * 0.00035)))
    height, width = mask.shape
    candidates: list[tuple[np.ndarray, dict[str, Any]]] = []

    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if touches_border or area < minimum_area:
            continue

        component = np.where(labels == index, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        aspect = float(w / h) if h else 0.0
        area_fraction = float(area / foreground_area)
        if area_fraction >= 0.08:
            subtype = "structural_opening"
        elif circularity >= 0.68 and 0.65 <= aspect <= 1.54:
            subtype = "circular_aperture"
        else:
            subtype = "decorative_openwork"
        candidates.append(
            (
                component,
                {
                    "center_px": {
                        "x": round(float(centroids[index][0]), 3),
                        "y": round(float(centroids[index][1]), 3),
                    },
                    "bbox_px": {"x": x, "y": y, "width": w, "height": h},
                    "area_px2": area,
                    "area_fraction_of_foreground": round(area_fraction, 8),
                    "circularity": round(circularity, 6),
                    "aspect_ratio": round(aspect, 6),
                    "candidate_type": "negative_space_candidate",
                    "candidate_subtype": subtype,
                    "semantic_state": "observable_geometry_semantics_unresolved",
                    "is_confirmed_stone": False,
                    "review_state": "pending",
                },
            )
        )

    candidates.sort(key=lambda item: (item[1]["center_px"]["y"], item[1]["center_px"]["x"]))
    return candidates


def _overlay(
    source: np.ndarray, candidates: list[tuple[np.ndarray, dict[str, Any]]]
) -> np.ndarray:
    canvas = source.copy()
    palette = ((48, 182, 232), (206, 115, 42), (181, 64, 187))
    for index, (mask, record) in enumerate(candidates, start=1):
        color = palette[(index - 1) % len(palette)]
        active = mask > 0
        canvas[active] = (
            canvas[active].astype(np.float32) * 0.35
            + np.asarray(color, dtype=np.float32) * 0.65
        ).astype(np.uint8)
        x = int(record["bbox_px"]["x"])
        y = int(record["bbox_px"]["y"])
        cv2.putText(
            canvas,
            str(index),
            (x, max(16, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _contact_sheet(tiles: list[tuple[str, np.ndarray, int]]) -> np.ndarray:
    tile_size = 230
    rows: list[np.ndarray] = []
    for row_start in range(0, len(tiles), 4):
        row_tiles = []
        for label, image, count in tiles[row_start : row_start + 4]:
            view = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            panel = cv2.copyMakeBorder(
                view, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(248, 248, 248)
            )
            cv2.putText(
                panel,
                f"{label.upper()}  |  {count} CANDIDATES",
                (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (24, 24, 24),
                1,
                cv2.LINE_AA,
            )
            row_tiles.append(panel)
        while len(row_tiles) < 4:
            row_tiles.append(np.full_like(row_tiles[0], 248))
        rows.append(cv2.hconcat(row_tiles))
    return cv2.vconcat(rows)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = root / "dataset/prepared_v1/manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: dict[str, Any] = {}
    front_tiles: list[tuple[str, np.ndarray, int]] = []
    total_candidates = 0
    subtype_counts = {
        "structural_opening": 0,
        "circular_aperture": 0,
        "decorative_openwork": 0,
    }

    for source_record in records:
        object_id = source_record["object_id"]
        object_views: dict[str, Any] = {}
        for view in VIEWS:
            view_record = source_record["views"][view]
            source = _read(root / view_record["image_path"], cv2.IMREAD_COLOR)
            mask = (_read(root / view_record["jewelry_mask_path"], cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) * 255
            candidates = _enclosed_components(mask)
            view_candidates = []
            combined = np.zeros_like(mask)
            for candidate_index, (candidate_mask, candidate) in enumerate(candidates, start=1):
                candidate_id = f"{object_id}_{view}_negative_space_{candidate_index:03d}"
                candidate_path = output_dir / "masks" / object_id / view / f"{candidate_id}.png"
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(candidate_path), candidate_mask):
                    raise OSError(f"Could not write {candidate_path}")
                combined = cv2.bitwise_or(combined, candidate_mask)
                item = {
                    "candidate_id": candidate_id,
                    **candidate,
                    "mask_path": _relative(candidate_path, root),
                    "mask_sha256": _sha256(candidate_path),
                }
                view_candidates.append(item)
                subtype_counts[item["candidate_subtype"]] += 1
            overlay = _overlay(source, candidates)
            overlay_path = output_dir / "overlays" / object_id / f"{view}.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise OSError(f"Could not write {overlay_path}")
            object_views[view] = {
                "candidate_count": len(view_candidates),
                "candidates": view_candidates,
                "overlay_path": _relative(overlay_path, root),
                "pairwise_disjoint": bool(np.count_nonzero(combined) == sum(item["area_px2"] for item in view_candidates)),
                "cross_view_identity_resolved": False,
            }
            total_candidates += len(view_candidates)
            if view == "front":
                front_tiles.append((object_id, overlay, len(view_candidates)))
        objects[object_id] = {
            "split": source_record["split"],
            "views": object_views,
            "confirmed_stone_count": 0,
            "component_semantics_reviewed": False,
        }

    audit_path = output_dir / "stl1_front_component_candidate_audit.png"
    if not cv2.imwrite(str(audit_path), _contact_sheet(front_tiles)):
        raise OSError(f"Could not write {audit_path}")

    checks = {
        "twenty_four_objects_processed": len(objects) == 24,
        "five_views_per_object_processed": all(len(item["views"]) == 5 for item in objects.values()),
        "candidate_masks_are_disjoint_per_view": all(
            view["pairwise_disjoint"]
            for item in objects.values()
            for view in item["views"].values()
        ),
        "no_unlabelled_opening_promoted_to_stone": all(
            not candidate["is_confirmed_stone"]
            for item in objects.values()
            for view in item["views"].values()
            for candidate in view["candidates"]
        ),
    }
    report = {
        "schema_version": "lalitha_phase4_stl1_component_candidates_v1",
        "stage": "Bible Phase 4 dataset-wide component candidate inventory",
        "authority": "machine_image_evidence_review_required",
        "source_manifest": _relative(manifest_path, root),
        "object_count": len(objects),
        "view_count": len(objects) * len(VIEWS),
        "candidate_count": total_candidates,
        "candidate_subtype_counts": subtype_counts,
        "objects": objects,
        "checks": checks,
        "gate": {
            "dataset_wide_negative_space_inventory_available": all(checks.values()),
            "individual_stones_confirmed": False,
            "cross_view_component_identity_resolved": False,
            "component_graph_generation_authorized": False,
            "reason": "STL-1 contains monochrome CAD-style renders without component labels; enclosed image regions establish negative-space evidence but do not identify material or prove gemstones.",
        },
        "claims": {
            "individual_stone_segmentation_complete": False,
            "stone_count_known": False,
            "stone_seats_or_cavities_known": False,
            "manufacturing_geometry_recovered": False,
        },
        "limitations": [
            "A circular opening may be an aperture, openwork, bezel interior or missing insert; image appearance alone does not establish gemstone material.",
            "Candidate IDs are view-local. Cross-view correspondence has not been established.",
            "No physical scale, calibrated cameras, component labels or CAD targets are available for STL-1.",
        ],
        "artifacts": {"front_audit": _relative(audit_path, root)},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase4_stl1_component_candidates_v1"),
    )
    args = parser.parse_args()
    report = run(args.root.resolve(), (args.root / args.output_dir).resolve())
    print(json.dumps({
        "object_count": report["object_count"],
        "view_count": report["view_count"],
        "candidate_count": report["candidate_count"],
        "checks": report["checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
