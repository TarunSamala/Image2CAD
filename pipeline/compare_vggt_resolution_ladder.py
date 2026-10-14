"""Compare retained VGGT resolutions without promoting any run to valid CAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


RESOLUTIONS = {
    224: "vggt_multiring_224_v1",
    280: "vggt_resolution_280_v1",
    336: "vggt_resolution_336_v1",
    448: "vggt_resolution_448_v1",
    518: "vggt_resolution_518_v1",
}
OBJECTS = ("ring_001", "ring_007", "ring_023")
VALIDATION_OBJECTS = ("ring_004", "ring_012", "ring_020")
VALIDATION_RUNS = {
    448: "vggt_resolution_448_validation_v1",
    518: "vggt_resolution_518_validation_v1",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(root: Path, resolution: int, directory: str) -> dict[str, Any]:
    run_root = root / "dataset/phase_runs/v1" / directory
    run = _read(run_root / "run_report.json")
    validation = _read(run_root / "validation_report.json")
    objects = {item["object_id"]: item for item in validation["objects"]}
    runtimes = {item["object_id"]: item for item in run["objects"]}
    complete = set(objects) == set(OBJECTS)
    heldout = [objects[object_id] for object_id in OBJECTS[1:] if object_id in objects]
    return {
        "resolution": resolution,
        "directory": directory,
        "object_count": len(objects),
        "complete_representative_cohort": complete,
        "peak_cuda_gib": round(
            max(item["peak_cuda_bytes"] for item in runtimes.values()) / 1024**3,
            6,
        ),
        "mean_elapsed_seconds": round(
            float(np.mean([item["elapsed_seconds"] for item in runtimes.values()])),
            6,
        ),
        "objects": objects,
        "heldout": {
            "object_count": len(heldout),
            "mean_iou": round(
                float(np.mean([item["mean_leave_one_view_out_iou"] for item in heldout])),
                6,
            ) if heldout else None,
            "mean_precision": round(
                float(np.mean([item["mean_partial_surface_precision"] for item in heldout])),
                6,
            ) if heldout else None,
            "mean_recall": round(
                float(np.mean([item["mean_surface_completion_recall"] for item in heldout])),
                6,
            ) if heldout else None,
            "minimum_view_iou": round(
                min(item["minimum_leave_one_view_out_iou"] for item in heldout),
                6,
            ) if heldout else None,
        },
    }


def _validation_summary(root: Path, resolution: int, directory: str) -> dict[str, Any]:
    run_root = root / "dataset/phase_runs/v1" / directory
    run = _read(run_root / "run_report.json")
    validation = _read(run_root / "validation_report.json")
    objects = {item["object_id"]: item for item in validation["objects"]}
    runtimes = {item["object_id"]: item for item in run["objects"]}
    return {
        "resolution": resolution,
        "directory": directory,
        "object_ids": list(objects),
        "all_objects_from_validation_split": all(
            item["split"] == "val" for item in objects.values()
        ),
        "complete_validation_cohort": set(objects) == set(VALIDATION_OBJECTS),
        "mean_iou": round(
            float(np.mean([item["mean_leave_one_view_out_iou"] for item in objects.values()])),
            6,
        ),
        "mean_precision": round(
            float(np.mean([item["mean_partial_surface_precision"] for item in objects.values()])),
            6,
        ),
        "mean_recall": round(
            float(np.mean([item["mean_surface_completion_recall"] for item in objects.values()])),
            6,
        ),
        "minimum_view_iou": round(
            min(item["minimum_leave_one_view_out_iou"] for item in objects.values()),
            6,
        ),
        "peak_cuda_gib": round(
            max(item["peak_cuda_bytes"] for item in runtimes.values()) / 1024**3,
            6,
        ),
        "objects": objects,
    }


def _plot(
    rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    colours = {"ring_001": "#111827", "ring_007": "#d84a78", "ring_023": "#747b84"}
    fields = (
        ("mean_leave_one_view_out_iou", "Surface-completion IoU"),
        ("mean_partial_surface_precision", "Partial-surface precision"),
        ("mean_surface_completion_recall", "Completion recall"),
    )
    for axis, (field, title) in zip(axes.flat[:3], fields):
        for object_id in OBJECTS:
            points = [
                (row["resolution"], row["objects"][object_id][field])
                for row in rows
                if object_id in row["objects"]
            ]
            axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker="o",
                linewidth=2,
                label=object_id,
                color=colours[object_id],
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Input resolution (px)")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    complete = [row for row in rows if row["complete_representative_cohort"]]
    axis = axes.flat[3]
    axis.plot(
        [row["resolution"] for row in complete],
        [row["heldout"]["mean_iou"] for row in complete],
        marker="o",
        linewidth=2,
        color="#d84a78",
        label="Held-out mean IoU",
    )
    axis.plot(
        [row["resolution"] for row in validation_rows],
        [row["mean_iou"] for row in validation_rows],
        marker="^",
        linewidth=2,
        color="#111827",
        label="Validation mean IoU",
    )
    memory_axis = axis.twinx()
    memory_axis.plot(
        [row["resolution"] for row in rows],
        [row["peak_cuda_gib"] for row in rows],
        marker="s",
        linestyle="--",
        color="#747b84",
        label="Peak CUDA GiB",
    )
    axis.set_title("Held-out gain and runtime memory", loc="left", fontweight="bold")
    axis.set_xlabel("Input resolution (px)")
    axis.set_ylim(0, 1)
    memory_axis.set_ylim(0, 4)
    axis.grid(alpha=0.2)
    lines = axis.lines + memory_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False)
    figure.suptitle("VGGT RESOLUTION LADDER · STL-1 REPRESENTATIVE COHORT", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _visual_comparison(root: Path, output: Path) -> None:
    columns = []
    for resolution in (224, 518):
        rows = []
        for object_id in OBJECTS:
            source = (
                root
                / "dataset/phase_runs/v1"
                / RESOLUTIONS[resolution]
                / object_id
                / f"{object_id}_cross_view_reprojection.png"
            )
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(source)
            width = 520
            cell_height = 780
            scale = min(width / image.shape[1], cell_height / image.shape[0])
            resized_width = int(round(image.shape[1] * scale))
            resized_height = int(round(image.shape[0] * scale))
            resized = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
            image = np.full((cell_height, width, 3), 248, dtype=np.uint8)
            x = (width - resized_width) // 2
            y = (cell_height - resized_height) // 2
            image[y : y + resized_height, x : x + resized_width] = resized
            header = np.full((46, width, 3), 248, dtype=np.uint8)
            cv2.putText(
                header,
                f"{object_id} / {resolution} PX",
                (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            rows.append(np.vstack((header, image)))
        columns.append(np.vstack(rows))
    if not cv2.imwrite(str(output), np.hstack(columns)):
        raise RuntimeError(f"Could not write {output}")


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    rows = [_summary(root, resolution, directory) for resolution, directory in RESOLUTIONS.items()]
    validation_rows = [
        _validation_summary(root, resolution, directory)
        for resolution, directory in VALIDATION_RUNS.items()
    ]
    eligible_validation = [
        row
        for row in validation_rows
        if row["complete_validation_cohort"] and row["peak_cuda_gib"] < 4.0
    ]
    selected_validation = max(eligible_validation, key=lambda row: row["mean_iou"])
    selected = next(
        row for row in rows if row["resolution"] == selected_validation["resolution"]
    )
    baseline = next(row for row in rows if row["resolution"] == 224)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart = output_dir / "vggt_resolution_ladder.png"
    comparison = output_dir / "vggt_224_vs_518_reprojection.jpg"
    _plot(rows, validation_rows, chart)
    _visual_comparison(root, comparison)
    report = {
        "schema_version": "vggt_resolution_ladder_v1",
        "dataset": "STL-1",
        "resolutions": rows,
        "validation_selection": {
            "object_ids": list(VALIDATION_OBJECTS),
            "candidates": validation_rows,
            "selected_resolution": selected_validation["resolution"],
            "selected_mean_iou": selected_validation["mean_iou"],
            "selection_rule": "highest mean IoU across three validation-split objects under 4 GiB peak CUDA allocation",
        },
        "selected_resolution": selected["resolution"],
        "selected_configuration": selected["directory"],
        "heldout_change_from_224": {
            "mean_iou": round(
                selected["heldout"]["mean_iou"] - baseline["heldout"]["mean_iou"],
                6,
            ),
            "mean_precision": round(
                selected["heldout"]["mean_precision"] - baseline["heldout"]["mean_precision"],
                6,
            ),
            "mean_recall": round(
                selected["heldout"]["mean_recall"] - baseline["heldout"]["mean_recall"],
                6,
            ),
        },
        "checks": {
            "selection_uses_validation_split_only": True,
            "all_selection_objects_are_validation_split": all(
                row["all_objects_from_validation_split"] for row in validation_rows
            ),
            "selected_run_below_four_gib": selected["peak_cuda_gib"] < 4.0,
            "selected_run_improves_heldout_mean_iou": selected["heldout"]["mean_iou"] > baseline["heldout"]["mean_iou"],
            "all_selected_objects_pass_geometry_gate": all(
                item["decision"] == "accepted_for_geometry_refinement"
                for item in selected["objects"].values()
            ),
        },
        "decision": "selected_as_research_baseline_not_validated_geometry",
        "artifacts": {
            "metric_chart": str(chart.relative_to(root)),
            "visual_comparison": str(comparison.relative_to(root)),
        },
        "claims": {
            "cross_view_geometry_validated": False,
            "hidden_surface_completion_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Resolution was selected on three STL-1 validation objects, not a broad external multi-view dataset.",
            "The test outputs were observed during the exploratory ladder before validation-based selection; a new external multi-view cohort is still required for a final unbiased estimate.",
            "Higher projected coverage does not infer surfaces absent from every source view.",
            "The selected 518 px run still fails the 0.80 mean and 0.70 minimum geometry gate.",
            "All coordinates remain relative and non-metric.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/vggt_resolution_ladder_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(json.dumps(run(root, output), indent=2))


if __name__ == "__main__":
    main()
