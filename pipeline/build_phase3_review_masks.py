"""Build physics-corrected Phase 3 evaluation silhouettes.

This does not create human ground truth.  It removes enclosed background
regions that cannot be negative space in an opaque solid.  In particular,
elongated highlights inside the visible band in front/top views are filled,
while ring openings and head/gallery cavities are preserved.  Every changed
region is recorded and raw machine-mask scores remain part of validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from phase3_objective import detail_regions, read_component_masks


VIEWS = ("front", "side", "top", "angled", "back")


def build(input_dir: Path, masks_dir: Path, output_dir: Path) -> dict:
    targets = read_component_masks(masks_dir, "ring01", VIEWS)
    regions = detail_regions(targets)
    mask_dir, review_dir = output_dir / "jewelry", output_dir / "review"
    mask_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    views = {}
    for view in VIEWS:
        raw = targets[view]["jewelry"]
        corrected = raw.copy()
        inverse = np.where(raw > 0, 0, 255).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse)
        x0, y0, x1, y1 = regions[view]
        changed, preserved = [], []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if x == 0 or y == 0 or x + width == raw.shape[1] or y + height == raw.shape[0]:
                continue
            component = labels == label
            in_detail = np.count_nonzero(component[y0 : y1 + 1, x0 : x1 + 1]) > 0
            record = {"area_px2": area, "bbox_px": [x, y, width, height], "overlaps_detail_region": bool(in_detail)}
            elongated_band_highlight = (
                view in {"front", "top"}
                and area < 1000
                and width >= 3 * max(height, 1)
            )
            if elongated_band_highlight:
                corrected[component] = 255
                changed.append(record | {"reason": "reviewed elongated reflection inside opaque band"})
            elif area >= 1000 or (in_detail and area >= 8):
                preserved.append(record | {"reason": "major opening" if area >= 1000 else "head/gallery negative space"})
            else:
                corrected[component] = 255
                changed.append(record | {"reason": "enclosed reflective/background artifact outside detail geometry"})
        mask_path = mask_dir / f"ring01_{view}_jewelry.png"
        cv2.imwrite(str(mask_path), corrected)
        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        cells = []
        for title, mask in (("RAW MACHINE MASK", raw), ("PHYSICS-CORRECTED REVIEW", corrected)):
            cell = image.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cell, contours, -1, (20, 30, 220), 1, cv2.LINE_AA)
            canvas = np.full((cell.shape[0] + 24, cell.shape[1], 3), 248, np.uint8)
            canvas[24:] = cell
            cv2.putText(canvas, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            cells.append(canvas)
        review_path = review_dir / f"ring01_{view}_mask_review.png"
        cv2.imwrite(str(review_path), np.hstack(cells))
        views[view] = {
            "filled_regions": changed, "preserved_regions": preserved,
            "raw_area_px2": int(np.count_nonzero(raw)), "corrected_area_px2": int(np.count_nonzero(corrected)),
            "mask": str(mask_path), "review": str(review_path),
        }
    report = {
        "stage": "phase3_physics_corrected_evaluation_masks", "human_ground_truth": False,
        "policy": "fill reviewed elongated band reflections in front/top plus enclosed non-detail artifacts below 1000 px; preserve major openings and head/gallery regions",
        "views": views,
    }
    (output_dir / "review_mask_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_3/evaluation_masks"))
    args = parser.parse_args()
    print(json.dumps(build(args.input_dir, args.masks_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
