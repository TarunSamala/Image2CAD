"""Run a bounded, research-only VGGT camera experiment on Ring01 images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.vggt_geometry_adapter import (
    NONCOMMERCIAL_CHECKPOINT_LICENSE,
    run_vggt_object,
)


VIEWS = ("front", "side", "top", "angled", "back")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/ring01_vggt_low_vram_v1"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--views", nargs="*", choices=VIEWS, default=list(VIEWS))
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run camera, depth and world-point heads instead of camera-only mode",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    images = [
        root / "data" / "ring01_reference_images" / f"ring01_{view}.png"
        for view in args.views
    ]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Ring01 inputs: {missing}")

    report = run_vggt_object(
        images,
        output,
        args.checkpoint.resolve(),
        NONCOMMERCIAL_CHECKPOINT_LICENSE,
        device="cuda",
        save_dense=args.full,
        allow_noncommercial_research=True,
        allow_low_vram_attempt=True,
        camera_only=not args.full,
        image_size=args.image_size,
    )
    report["object_id"] = "ring01"
    report["decision"] = "research_only"
    report["claims"] = {
        "camera_pose_accuracy_validated": False,
        "metric_depth_validated": False,
        "geometry_accuracy_validated": False,
        "manufacturing_accuracy_validated": False,
    }
    (output / "vggt_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
