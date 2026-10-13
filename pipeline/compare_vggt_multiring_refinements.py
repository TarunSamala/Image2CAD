"""Compare full-frame and object-centric VGGT multi-ring experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resize_width(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _header(width: int, title: str, subtitle: str, height: int = 62) -> np.ndarray:
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 16, 16), 1, cv2.LINE_AA)
    cv2.putText(image, subtitle, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (65, 65, 65), 1, cv2.LINE_AA)
    return image


def run(root: Path, baseline_dir: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    baseline = _read(baseline_dir / "validation_report.json")
    candidate = _read(candidate_dir / "validation_report.json")
    baseline_by_id = {item["object_id"]: item for item in baseline["objects"]}
    candidate_by_id = {item["object_id"]: item for item in candidate["objects"]}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("The VGGT refinement cohorts differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    rows = []
    for object_id in baseline_by_id:
        original = baseline_by_id[object_id]
        refined = candidate_by_id[object_id]
        delta = (
            refined["mean_leave_one_view_out_iou"]
            - original["mean_leave_one_view_out_iou"]
        )
        objects.append(
            {
                "object_id": object_id,
                "split": original["split"],
                "full_frame_mean_iou": original["mean_leave_one_view_out_iou"],
                "object_centric_mean_iou": refined["mean_leave_one_view_out_iou"],
                "delta": round(delta, 6),
                "full_frame_decision": original["decision"],
                "object_centric_decision": refined["decision"],
            }
        )
        full_image = cv2.imread(
            str(baseline_dir / object_id / f"{object_id}_cross_view_reprojection.png")
        )
        refined_image = cv2.imread(
            str(candidate_dir / object_id / f"{object_id}_cross_view_reprojection.png")
        )
        if full_image is None or refined_image is None:
            raise FileNotFoundError(f"Missing comparison preview for {object_id}")
        full_image = _resize_width(full_image, 560)
        refined_image = _resize_width(refined_image, 560)
        baseline_label = _header(
            560,
            f"{object_id.upper()} / FULL FRAME",
            f"leave-one-view-out mean IoU {original['mean_leave_one_view_out_iou']:.3f}",
        )
        candidate_label = _header(
            560,
            "OBJECT-CENTRIC REFRAMING",
            f"mean IoU {refined['mean_leave_one_view_out_iou']:.3f} / delta {delta:+.3f}",
        )
        rows.append(
            np.hstack(
                (
                    np.vstack((baseline_label, full_image)),
                    np.vstack((candidate_label, refined_image)),
                )
            )
        )

    full_mean = float(np.mean([item["full_frame_mean_iou"] for item in objects]))
    candidate_mean = float(
        np.mean([item["object_centric_mean_iou"] for item in objects])
    )
    sheet = np.vstack(rows)
    preview = output_dir / "vggt_refinement_comparison.jpg"
    if not cv2.imwrite(str(preview), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write {preview}")
    report = {
        "schema_version": "vggt_multiring_refinement_comparison_v1",
        "dataset": "STL-1",
        "object_count": len(objects),
        "objects": objects,
        "cohort": {
            "full_frame_mean_iou": round(full_mean, 6),
            "object_centric_mean_iou": round(candidate_mean, 6),
            "delta": round(candidate_mean - full_mean, 6),
            "full_frame_accepted_count": baseline["accepted_object_count"],
            "object_centric_accepted_count": candidate["accepted_object_count"],
        },
        "decision": "object_centric_rejected_no_cohort_improvement",
        "selected_configuration": "prepared_v1_full_frame",
        "artifacts": {"comparison_preview": str(preview.relative_to(root))},
        "claims": {
            "cross_view_geometry_validated": False,
            "dataset_generalization_validated": False,
            "manufacturing_accuracy_validated": False,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_224_v1"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_objectcentric_224_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_multiring_refinement_comparison_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    print(
        json.dumps(
            run(
                root,
                resolve(args.baseline_dir),
                resolve(args.candidate_dir),
                resolve(args.output_dir),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
