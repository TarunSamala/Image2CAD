"""Extract learned VGGT cross-view tracks for Phase 3 correspondence research.

The pinned local checkpoint is used without network access.  Query points are
selected independently in each source silhouette, then tracked through all five
views.  Outputs are non-metric machine evidence and do not establish physical
camera calibration or geometry accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.vggt_geometry_adapter import PINNED_VGGT_COMMIT, _sha256


VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")
VALIDATION_OBJECTS = ("ring_004", "ring_012", "ring_020")
EVALUATION_OBJECTS = ("ring_001", "ring_007", "ring_015", "ring_023")
TRACK_OBJECTS = (*VALIDATION_OBJECTS, *EVALUATION_OBJECTS)
IMAGE_SIZE = 518


def _records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["object_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _square(path: Path, size: int, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    side = max(height, width)
    shape = (side, side) if grayscale else (side, side, 3)
    canvas = np.zeros(shape, dtype=np.uint8)
    y = (side - height) // 2
    x = (side - width) // 2
    canvas[y : y + height, x : x + width] = image
    interpolation = cv2.INTER_NEAREST if grayscale else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def _select_query_points(
    image: np.ndarray,
    mask: np.ndarray,
    count: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_u8 = np.uint8(mask) * 255
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=count,
        qualityLevel=0.005,
        minDistance=5,
        mask=mask_u8,
        blockSize=5,
        useHarrisDetector=False,
    )
    selected = [] if corners is None else corners[:, 0].tolist()
    occupied = {
        (int(round(point[0])), int(round(point[1]))) for point in selected
    }
    if len(selected) < count:
        ys, xs = np.nonzero(mask)
        if len(xs):
            order = np.linspace(0, len(xs) - 1, min(len(xs), count * 8), dtype=int)
            for index in order:
                point = (int(xs[index]), int(ys[index]))
                if point in occupied:
                    continue
                selected.append([float(point[0]), float(point[1])])
                occupied.add(point)
                if len(selected) == count:
                    break
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} valid query points for requested {count}")
    return np.asarray(selected[:count], dtype=np.float32)


def _track_object(
    root: Path,
    object_id: str,
    record: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    output_dir: Path,
    query_count: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square

    sources = [
        _square(root / record["views"][view]["image_path"], IMAGE_SIZE)
        for view in VIEW_ORDER
    ]
    masks = [
        _square(
            root / record["views"][view]["jewelry_mask_path"],
            IMAGE_SIZE,
            grayscale=True,
        )
        > 0
        for view in VIEW_ORDER
    ]
    query_points = np.stack(
        [
            _select_query_points(source, mask, query_count)
            for source, mask in zip(sources, masks)
        ]
    )
    image_paths = [root / record["views"][view]["image_path"] for view in VIEW_ORDER]
    started = time.perf_counter()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=True,
    ).half()
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Track-only VGGT is missing checkpoint keys: {incompatible.missing_keys[:8]}"
        )
    model = model.cuda().eval()
    images, _ = load_and_preprocess_images_square(
        [str(path) for path in image_paths], target_size=IMAGE_SIZE
    )
    images = images.unsqueeze(0).to(device="cuda", dtype=torch.float16)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        aggregated_tokens, patch_start_idx = model.aggregator(images)
        feature_maps = model.track_head.feature_extractor(
            aggregated_tokens,
            images,
            patch_start_idx,
        )

    model.aggregator = None
    model.track_head.feature_extractor = None
    aggregated_tokens = None
    images = None
    gc.collect()
    torch.cuda.empty_cache()

    tracks = np.zeros((len(VIEW_ORDER), len(VIEW_ORDER), query_count, 2), np.float32)
    visibility = np.zeros((len(VIEW_ORDER), len(VIEW_ORDER), query_count), np.float32)
    confidence = np.zeros_like(visibility)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for query_index in range(len(VIEW_ORDER)):
            order = [query_index] + [
                index for index in range(len(VIEW_ORDER)) if index != query_index
            ]
            inverse = np.argsort(order)
            ordered_features = feature_maps[:, order]
            for start in range(0, query_count, query_chunk_size):
                stop = min(start + query_chunk_size, query_count)
                queries = torch.from_numpy(query_points[query_index, start:stop])
                queries = queries.unsqueeze(0).to(
                    device="cuda", dtype=torch.float16
                )
                coordinate_list, visible, certain = model.track_head.tracker(
                    query_points=queries,
                    fmaps=ordered_features,
                    iters=model.track_head.iters,
                )
                coordinates = coordinate_list[-1][0, inverse].float().cpu().numpy()
                visible_np = visible[0, inverse].float().cpu().numpy()
                certain_np = certain[0, inverse].float().cpu().numpy()
                tracks[query_index, :, start:stop] = coordinates
                visibility[query_index, :, start:stop] = visible_np
                confidence[query_index, :, start:stop] = certain_np
                del queries, coordinate_list, visible, certain
            del ordered_features

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "vggt_track_evidence.npz"
    np.savez_compressed(
        evidence_path,
        query_points_xy=query_points,
        tracks_xy=tracks,
        visibility=visibility,
        confidence=confidence,
    )
    in_bounds = (
        (tracks[..., 0] >= 0)
        & (tracks[..., 0] < IMAGE_SIZE)
        & (tracks[..., 1] >= 0)
        & (tracks[..., 1] < IMAGE_SIZE)
    )
    source_diagonal = np.arange(len(VIEW_ORDER))
    source_error = np.linalg.norm(
        tracks[source_diagonal, source_diagonal] - query_points,
        axis=-1,
    )
    report = {
        "schema_version": "vggt_track_evidence_v1",
        "object_id": object_id,
        "split": record["split"],
        "views": list(VIEW_ORDER),
        "image_size": IMAGE_SIZE,
        "query_count_per_view": query_count,
        "query_selection": "shi_tomasi_inside_silhouette_then_deterministic_mask_fill",
        "tracker": {
            "backend": "VGGT-1B track head",
            "pinned_code_commit": PINNED_VGGT_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "query_chunk_size": query_chunk_size,
            "dtype": "float16",
        },
        "summary": {
            "mean_visibility": round(float(np.mean(visibility)), 6),
            "mean_confidence": round(float(np.mean(confidence)), 6),
            "in_bounds_fraction": round(float(np.mean(in_bounds)), 6),
            "maximum_source_query_error_pixels": round(float(np.max(source_error)), 6),
        },
        "runtime": {
            "seconds": round(time.perf_counter() - started, 3),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "artifacts": {
            "track_evidence": str(evidence_path.relative_to(root)),
        },
        "claims": {
            "tracks_are_human_ground_truth": False,
            "camera_calibration_validated": False,
            "metric_scale_validated": False,
            "geometry_accuracy_validated": False,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    model = None
    feature_maps = None
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run(
    root: Path,
    checkpoint: Path,
    output_root: Path,
    object_ids: list[str],
    query_count: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    checkpoint_sha256 = _sha256(checkpoint)
    results = []
    for object_id in object_ids:
        if object_id not in TRACK_OBJECTS:
            raise ValueError(f"Object is outside the fixed Phase 3 cohort: {object_id}")
        object_output = output_root / object_id
        existing_report = object_output / "report.json"
        existing_evidence = object_output / "vggt_track_evidence.npz"
        if existing_report.is_file() and existing_evidence.is_file():
            retained = json.loads(existing_report.read_text(encoding="utf-8"))
            if (
                retained.get("query_count_per_view") == query_count
                and retained.get("tracker", {}).get("checkpoint_sha256")
                == checkpoint_sha256
            ):
                results.append(retained)
                continue
        results.append(
            _track_object(
                root,
                object_id,
                records[object_id],
                checkpoint,
                checkpoint_sha256,
                object_output,
                query_count,
                query_chunk_size,
            )
        )
    report = {
        "schema_version": "vggt_track_evidence_batch_v1",
        "dataset": "STL-1",
        "object_ids": object_ids,
        "object_count": len(results),
        "objects": results,
        "checks": {
            "all_objects_completed": len(results) == len(object_ids),
            "all_tracks_in_bounds_above_95_percent": all(
                item["summary"]["in_bounds_fraction"] >= 0.95 for item in results
            ),
            "all_source_queries_preserved": all(
                item["summary"]["maximum_source_query_error_pixels"] <= 0.01
                for item in results
            ),
        },
        "claims": {
            "track_accuracy_validated": False,
            "physical_correspondence_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/vggt/VGGT-1B/model.pt")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_track_evidence_v1"),
    )
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--query-count", type=int, default=96)
    parser.add_argument("--query-chunk-size", type=int, default=24)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    object_ids = args.object_ids or list(TRACK_OBJECTS)
    report = run(
        root,
        checkpoint,
        output_root,
        object_ids,
        args.query_count,
        args.query_chunk_size,
    )
    print(
        json.dumps(
            {
                "object_count": report["object_count"],
                "checks": report["checks"],
                "objects": [
                    {
                        "object_id": item["object_id"],
                        "summary": item["summary"],
                        "runtime": item["runtime"],
                    }
                    for item in report["objects"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
