"""Run the low-VRAM VGGT experiment on representative STL-1 rings.

The default cohort spans sculptural detail and held-out objects.  It is small
enough for the 4 GiB laptop GPU, while keeping the command generic so all 24
five-view objects can be evaluated later without changing code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.vggt_geometry_adapter import (
    NONCOMMERCIAL_CHECKPOINT_LICENSE,
    run_vggt_object,
)


VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")
DEFAULT_OBJECTS = ("ring_001", "ring_007", "ring_023")


def _records(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["object_id"]: row for row in rows}


def _object_centric_inputs(
    root: Path,
    record: dict[str, Any],
    output_dir: Path,
    image_size: int,
    margin_fraction: float,
) -> list[Path]:
    """Create versioned object-centric views without changing source evidence."""

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    manifest: dict[str, Any] = {
        "schema_version": "vggt_object_centric_inputs_v1",
        "object_id": record["object_id"],
        "image_size": image_size,
        "margin_fraction": margin_fraction,
        "views": {},
    }
    for view in VIEW_ORDER:
        source_path = root / record["views"][view]["image_path"]
        mask_path = root / record["views"][view]["jewelry_mask_path"]
        if not source_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared image or mask for {record['object_id']} {view}"
            )
        source = Image.open(source_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_array = np.asarray(mask)
        rows, columns = np.nonzero(mask_array > 0)
        if not len(rows):
            raise ValueError(f"Empty jewellery mask for {record['object_id']} {view}")
        x = int(columns.min())
        y = int(rows.min())
        width = int(columns.max() - x + 1)
        height = int(rows.max() - y + 1)
        crop_side = int(round(max(width, height) * (1.0 + 2.0 * margin_fraction)))
        crop_side = min(crop_side, min(source.size))
        centre_x = x + width / 2.0
        centre_y = y + height / 2.0
        x0 = int(round(centre_x - crop_side / 2.0))
        y0 = int(round(centre_y - crop_side / 2.0))
        x0 = min(max(x0, 0), source.width - crop_side)
        y0 = min(max(y0, 0), source.height - crop_side)
        box = (x0, y0, x0 + crop_side, y0 + crop_side)
        normalized = source.crop(box).resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        normalized_mask = mask.crop(box).resize(
            (image_size, image_size), Image.Resampling.NEAREST
        )
        image_output = inputs_dir / f"{view}.png"
        mask_output = inputs_dir / f"{view}_mask.png"
        normalized.save(image_output)
        normalized_mask.save(mask_output)
        paths.append(image_output)
        manifest["views"][view] = {
            "source_image": str(source_path.relative_to(root)),
            "source_mask": str(mask_path.relative_to(root)),
            "image": str(image_output.relative_to(root)),
            "mask": str(mask_output.relative_to(root)),
            "source_foreground_bbox_xywh": [x, y, width, height],
            "crop_xyxy": [x0, y0, x0 + crop_side, y0 + crop_side],
        }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    records = _records(root / "dataset/prepared_v1/manifest.jsonl")
    unknown = set(args.objects) - set(records)
    if unknown:
        raise ValueError(f"Unknown STL-1 object ids: {sorted(unknown)}")

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for object_id in args.objects:
        record = records[object_id]
        object_dir = output_root / object_id
        if (object_dir / "vggt_report.json").exists() and not args.force:
            raise FileExistsError(
                f"{object_dir / 'vggt_report.json'} exists; use --force to rerun"
            )
        images = (
            _object_centric_inputs(
                root,
                record,
                object_dir,
                args.image_size,
                args.object_margin_fraction,
            )
            if args.object_centric
            else [root / record["views"][view]["image_path"] for view in VIEW_ORDER]
        )
        report = run_vggt_object(
            images,
            object_dir,
            args.checkpoint.resolve(),
            NONCOMMERCIAL_CHECKPOINT_LICENSE,
            device="cuda",
            save_dense=True,
            allow_noncommercial_research=True,
            allow_low_vram_attempt=True,
            camera_only=False,
            image_size=args.image_size,
        )
        report.update(
            {
                "object_id": object_id,
                "dataset": "STL-1",
                "split": record["split"],
                "view_order": list(VIEW_ORDER),
                "input_preprocessing": (
                    "object_centric_mask_bbox"
                    if args.object_centric
                    else "prepared_v1_full_frame"
                ),
                "decision": "research_only",
                "claims": {
                    "camera_pose_accuracy_validated": False,
                    "metric_depth_validated": False,
                    "cross_view_geometry_validated": False,
                    "manufacturing_accuracy_validated": False,
                },
            }
        )
        (object_dir / "vggt_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        results.append(
            {
                "object_id": object_id,
                "split": record["split"],
                "status": report["status"],
                "peak_cuda_bytes": report["peak_cuda_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
                "report": str((object_dir / "vggt_report.json").relative_to(root)),
            }
        )
        print(f"VGGT {object_id}: {report['status']}", flush=True)

    summary = {
        "schema_version": "vggt_multiring_low_vram_run_v1",
        "dataset": "STL-1",
        "object_count": len(results),
        "view_count": len(results) * len(VIEW_ORDER),
        "image_size": args.image_size,
        "input_preprocessing": (
            "object_centric_mask_bbox"
            if args.object_centric
            else "prepared_v1_full_frame"
        ),
        "objects": results,
        "selection_basis": {
            "ring_001": "dense face and sculptural detail",
            "ring_007": "complex held-out test object",
            "ring_023": "different held-out floral geometry",
        },
        "decision": "research_only",
        "claims": {
            "cross_view_geometry_validated": False,
            "dataset_generalization_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    (output_root / "run_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_224_v1"),
    )
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--object-centric", action="store_true")
    parser.add_argument("--object-margin-fraction", type=float, default=0.12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
