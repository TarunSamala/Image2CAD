"""Prepare the available five-view jewellery renders for research experiments.

The source directory is never modified. The prepared dataset contains
standardized images, conservative foreground masks, edge maps, object-level
splits, and manifests which explicitly record unavailable supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("front", "top", "iso", "lsv", "rsv")
VALIDATION_IDS = {4, 12, 20}
TEST_IDS = {7, 15, 23}


def _ring_number(path: Path) -> int:
    match = re.fullmatch(r"Ring\s+(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid ring directory name: {path}")
    return int(match.group(1))


def _view_name(path: Path) -> str:
    name = path.stem.lower()
    if "front" in name:
        return "front"
    if "top" in name:
        return "top"
    if "iso" in name:
        return "iso"
    if "lsv" in name:
        return "lsv"
    if "rsv" in name:
        return "rsv"
    raise ValueError(f"cannot identify view from filename: {path}")


def _split(number: int) -> str:
    if number in VALIDATION_IDS:
        return "val"
    if number in TEST_IDS:
        return "test"
    return "train"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _background_bgr(image: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        (
            image[:20].reshape(-1, 3),
            image[-20:].reshape(-1, 3),
            image[:, :20].reshape(-1, 3),
            image[:, -20:].reshape(-1, 3),
        )
    )
    return np.asarray(
        [np.bincount(border[:, channel], minlength=256).argmax() for channel in range(3)],
        dtype=np.uint8,
    )


def _foreground_mask(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(image.astype(np.float32) - background.astype(np.float32), axis=2)
    initial = np.where(distance > 18.0, 255, 0).astype(np.uint8)
    initial = cv2.morphologyEx(initial, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(initial)
    if count <= 1:
        raise ValueError("foreground extraction found no object")

    image_area = image.shape[0] * image.shape[1]
    candidates = [
        label for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= max(100, image_area * 0.0001)
    ]
    if not candidates:
        raise ValueError("foreground extraction found no substantial component")
    largest = max(candidates, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    x, y, width, height, _ = stats[largest]
    margin = max(width, height) * 0.20
    x0, y0, x1, y1 = x - margin, y - margin, x + width + margin, y + height + margin

    keep = []
    for label in candidates:
        area = stats[label, cv2.CC_STAT_AREA]
        center_x, center_y = centroids[label]
        near_object = x0 <= center_x <= x1 and y0 <= center_y <= y1
        substantial = area >= image_area * 0.002
        watermark_zone = center_x > image.shape[1] * 0.78 and center_y > image.shape[0] * 0.82
        if label == largest or ((near_object or substantial) and not watermark_zone):
            keep.append(label)
    mask = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("empty foreground mask")
    return tuple(int(value) for value in cv2.boundingRect(points))


def _normalized(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x, y, width, height = _bbox(mask)
    padding = int(round(max(width, height) * 0.10))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(image.shape[1], x + width + padding), min(image.shape[0], y + height + padding)
    crop_image, crop_mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    margin = max(24, size // 16)
    scale = min((size - 2 * margin) / crop_image.shape[1], (size - 2 * margin) / crop_image.shape[0])
    target_width = max(1, int(round(crop_image.shape[1] * scale)))
    target_height = max(1, int(round(crop_image.shape[0] * scale)))
    resized_image = cv2.resize(crop_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((size, size, 3), background, dtype=np.uint8)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    offset_x, offset_y = (size - target_width) // 2, (size - target_height) // 2
    canvas[offset_y:offset_y + target_height, offset_x:offset_x + target_width] = resized_image
    mask_canvas[offset_y:offset_y + target_height, offset_x:offset_x + target_width] = resized_mask
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120)
    edges = cv2.bitwise_and(edges, cv2.dilate(mask_canvas, np.ones((3, 3), np.uint8)))
    transform = {
        "source_foreground_bbox_xywh": [x, y, width, height],
        "source_crop_xyxy": [x0, y0, x1, y1],
        "resize_scale": round(scale, 8),
        "placement_xywh": [offset_x, offset_y, target_width, target_height],
    }
    return canvas, mask_canvas, edges, transform


def prepare(source_dir: Path, output_dir: Path, image_size: int = 768) -> dict:
    ring_dirs = sorted(
        (path for path in source_dir.iterdir() if path.is_dir()),
        key=_ring_number,
    )
    if len(ring_dirs) != 24:
        raise ValueError(f"expected 24 ring directories, found {len(ring_dirs)}")

    records = []
    source_hashes: list[str] = []
    split_ids = {split: [] for split in ("train", "val", "test")}
    mask_occupancies = []
    for ring_dir in ring_dirs:
        number = _ring_number(ring_dir)
        object_id = f"ring_{number:03d}"
        split = _split(number)
        split_ids[split].append(object_id)
        files = {_view_name(path): path for path in ring_dir.glob("*.png")}
        if set(files) != set(VIEWS) or len(list(ring_dir.glob("*.png"))) != len(VIEWS):
            raise ValueError(f"{ring_dir} must contain exactly one image for each of {VIEWS}")

        record = {
            "schema_version": "jewellery_multiview_v1",
            "object_id": object_id,
            "source_object_name": ring_dir.name,
            "category": "ring",
            "split": split,
            "views": {},
            "supervision": {
                "jewelry_silhouette": "pseudo_background_difference",
                "edge_map": "opencv_canny",
                "individual_stones": None,
                "metal_components": None,
                "prongs": None,
                "stone_seats": None,
                "cavities": None,
                "camera_intrinsics": None,
                "camera_extrinsics": None,
                "physical_scale": None,
                "mesh_target": None,
                "cad_target": None,
            },
        }
        for view in VIEWS:
            source_path = files[view]
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode {source_path}")
            background = _background_bgr(image)
            mask = _foreground_mask(image, background)
            normalized, normalized_mask, edges, transform = _normalized(
                image, mask, background, image_size,
            )
            image_path = output_dir / "images" / object_id / f"{view}.png"
            mask_path = output_dir / "masks" / "jewelry" / object_id / f"{view}.png"
            edge_path = output_dir / "edges" / object_id / f"{view}.png"
            for path in (image_path, mask_path, edge_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), normalized)
            cv2.imwrite(str(mask_path), normalized_mask)
            cv2.imwrite(str(edge_path), edges)
            source_hash = _sha256(source_path)
            source_hashes.append(source_hash)
            occupancy = float(np.count_nonzero(normalized_mask) / normalized_mask.size)
            mask_occupancies.append(occupancy)
            record["views"][view] = {
                "source_path": str(source_path),
                "image_path": str(image_path),
                "jewelry_mask_path": str(mask_path),
                "edge_path": str(edge_path),
                "source_sha256": source_hash,
                "source_size_wh": [image.shape[1], image.shape[0]],
                "prepared_size_wh": [image_size, image_size],
                "background_rgb": [int(background[2]), int(background[1]), int(background[0])],
                "mask_occupancy": round(occupancy, 8),
                "transform": transform,
                "camera_semantic": view,
                "camera_calibrated": False,
            }
        records.append(record)
        object_path = output_dir / "objects" / f"{object_id}.json"
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, object_ids in split_ids.items():
        (split_dir / f"{split}.txt").write_text("\n".join(object_ids) + "\n", encoding="utf-8")

    checks = {
        "twenty_four_objects": len(records) == 24,
        "five_views_per_object": all(set(record["views"]) == set(VIEWS) for record in records),
        "all_source_images_unique": len(source_hashes) == len(set(source_hashes)),
        "object_splits_disjoint": not (
            set(split_ids["train"]) & set(split_ids["val"])
            or set(split_ids["train"]) & set(split_ids["test"])
            or set(split_ids["val"]) & set(split_ids["test"])
        ),
        "all_masks_nonempty": min(mask_occupancies) > 0.01,
        "no_unavailable_ground_truth_fabricated": all(
            record["supervision"][key] is None
            for record in records
            for key in ("individual_stones", "metal_components", "prongs", "stone_seats", "cavities", "mesh_target", "cad_target")
        ),
    }
    report = {
        "schema_version": "jewellery_multiview_v1",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "object_count": len(records),
        "image_count": len(records) * len(VIEWS),
        "views": list(VIEWS),
        "image_size": image_size,
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "mask_occupancy_min_mean_max": [
            round(min(mask_occupancies), 8),
            round(float(np.mean(mask_occupancies)), 8),
            round(max(mask_occupancies), 8),
        ],
        "checks": checks,
        "dataset_valid": all(checks.values()),
        "training_readiness": {
            "multi_view_vision_pretraining": True,
            "silhouette_and_edge_experiments": True,
            "supervised_image_to_cad": False,
            "quantitative_3d_evaluation": False,
        },
        "limitations": [
            "Only 24 independent objects are available; 120 views are not 120 independent samples.",
            "No mesh, CAD program, physical scale, camera calibration, or component ground truth is available.",
            "Pseudo-silhouettes require manual review before being treated as evaluation ground truth.",
            "All samples are blue renders of rings, so material and jewellery-category diversity is limited.",
            "Dataset provenance and redistribution licence are not recorded.",
        ],
    }
    (output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("dataset/STL-1"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/prepared_v1"))
    parser.add_argument("--image-size", type=int, default=768)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output_dir, args.image_size), indent=2))


if __name__ == "__main__":
    main()
