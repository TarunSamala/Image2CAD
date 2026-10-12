"""Prepare and evaluate the image-only Phase 2 metric-depth consensus experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.infer_metric_depth_backend import BACKEND_MODELS, sha256


RING01_VIEW_ORDER = ("front", "side", "top", "angled", "back")
STL1_VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")
DEFAULT_STL1_OBJECTS = ("ring_001", "ring_004", "ring_007")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        head = root / ".git" / "HEAD"
        try:
            value = head.read_text(encoding="utf-8").strip()
            if not value.startswith("ref: "):
                return value or None
            ref = root / ".git" / value.removeprefix("ref: ")
            return ref.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def robust_normalize(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    selected = values[mask]
    low, high = np.percentile(selected, (2, 98))
    if high <= low:
        high = low + 1e-6
    result = np.clip((values - low) / (high - low), 0, 1)
    return result.astype(np.float32), float(low), float(high)


def masked_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    x = left[mask].astype(np.float64)
    y = right[mask].astype(np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _colorize(normalized: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = np.round(np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~mask] = (242, 242, 242)
    return color


def colorize_depth(normalized_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render metric depth with the project convention: red near, blue far."""

    return _colorize(1.0 - np.clip(normalized_depth, 0, 1), mask)


def colorize_magnitude(normalized_magnitude: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render an error magnitude with blue low and red high."""

    return _colorize(normalized_magnitude, mask)


def panel(
    image: np.ndarray,
    title: str,
    width: int = 210,
    legend: str | None = None,
) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    header_height = 52
    canvas = np.full((resized.shape[0] + header_height, width, 3), 246, np.uint8)
    canvas[header_height:] = resized
    cv2.putText(
        canvas,
        title,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (24, 31, 40),
        1,
        cv2.LINE_AA,
    )
    if legend:
        if legend not in {"depth", "magnitude"}:
            raise ValueError(f"Unknown panel legend: {legend}")
        left_label, right_label = (
            ("NEAR", "FAR") if legend == "depth" else ("LOW", "HIGH")
        )
        gradient_width = max(1, width - 90)
        gradient = np.linspace(0.0, 1.0, gradient_width, dtype=np.float32)
        if legend == "depth":
            gradient = 1.0 - gradient
        gradient_u8 = np.round(gradient * 255).astype(np.uint8)[None]
        gradient_bgr = cv2.applyColorMap(gradient_u8, cv2.COLORMAP_TURBO)
        gradient_bgr = cv2.resize(
            gradient_bgr, (gradient_width, 8), interpolation=cv2.INTER_NEAREST
        )
        x0 = 45
        canvas[30:38, x0 : x0 + gradient_width] = gradient_bgr
        cv2.putText(
            canvas,
            left_label,
            (7, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (65, 65, 65),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            right_label,
            (width - 37, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (65, 65, 65),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_input_manifest(root: Path, output: Path, stl1_objects: tuple[str, ...]) -> dict[str, Any]:
    ring_manifest_path = root / "data" / "ring01_ground_truth_v1" / "manifest.json"
    ring_manifest = json.loads(ring_manifest_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for view in RING01_VIEW_ORDER:
        record = ring_manifest["views"][view]
        source = record["source"]
        mask = record["proposals"]["jewelry"]
        samples.append(
            {
                "sample_key": f"ring01_reference:{view}",
                "dataset": "Ring01",
                "object_id": "ring01_reference",
                "split": "reference",
                "view": view,
                "image_path": source["path"],
                "mask_path": mask["path"],
                "source_sha256": source["sha256"],
                "physical_scale": None,
                "camera_calibrated": False,
            }
        )

    for object_id in stl1_objects:
        object_path = root / "dataset" / "prepared_v1" / "objects" / f"{object_id}.json"
        record = json.loads(object_path.read_text(encoding="utf-8"))
        for view in STL1_VIEW_ORDER:
            view_record = record["views"][view]
            samples.append(
                {
                    "sample_key": f"{object_id}:{view}",
                    "dataset": "STL-1",
                    "object_id": object_id,
                    "split": record["split"],
                    "view": view,
                    "image_path": view_record["image_path"],
                    "mask_path": view_record["jewelry_mask_path"],
                    "source_sha256": sha256(root / view_record["image_path"]),
                    "physical_scale": None,
                    "camera_calibrated": False,
                }
            )

    manifest = {
        "schema_version": "image2cad_metric_depth_consensus_input_v1",
        "created_at_utc": utc_now(),
        "code_commit": git_commit(root),
        "execution_profile": "image_only_reconstruction_v1",
        "sample_count": len(samples),
        "object_count": 1 + len(stl1_objects),
        "selected_stl1_objects": list(stl1_objects),
        "selection_reason": {
            "ring_001": "train split representative",
            "ring_004": "validation split and high prior depth instability",
            "ring_007": "test split and high prior depth instability",
        },
        "samples": samples,
        "measurement_contract": {
            "pixel_measurements": "observed",
            "normalized_shape": "machine_hypothesis",
            "model_depth_unit": "metre_model_hypothesis",
            "absolute_millimetres": "blocked_without_known_dimension",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_backend(
    root: Path,
    experiment: Path,
    backend: str,
    expected_manifest_sha256: str,
) -> dict[str, Any] | None:
    path = experiment / "backends" / backend / "backend_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("backend") != backend:
        raise ValueError(f"Backend identity mismatch in {path}")
    if report.get("input_manifest_sha256") != expected_manifest_sha256:
        raise ValueError(f"Input manifest hash mismatch in {path}")
    for record in report["records"]:
        depth_path = root / record["artifacts"]["depth_m"]
        if not depth_path.is_file() or sha256(depth_path) != record["depth_sha256"]:
            raise ValueError(f"Depth hash mismatch for {record['sample_key']} in {backend}")
    return report


def aggregate(root: Path, experiment: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    available: dict[str, dict[str, Any]] = {}
    blocked: dict[str, dict[str, Any]] = {}
    incomplete: dict[str, dict[str, Any]] = {}
    manifest_sha256 = sha256(experiment / "input_manifest.json")
    for backend, model in BACKEND_MODELS.items():
        report = read_backend(root, experiment, backend, manifest_sha256)
        if report is None:
            blocked[backend] = {
                "status": "not_run",
                "model": model,
                "blocker": "backend_report_missing",
            }
        else:
            available[backend] = report
            if report["sample_count"] != manifest["sample_count"]:
                incomplete[backend] = {
                    "status": "partial_coverage",
                    "sample_count": report["sample_count"],
                    "requested_sample_count": manifest["sample_count"],
                    "missing_sample_count": manifest["sample_count"]
                    - report["sample_count"],
                    "reason": (
                        "cpu_memory_safety_probe_only_on_four_front_views"
                        if backend == "depth_pro"
                        else "backend_run_was_scoped"
                    ),
                }

    record_lookup = {
        backend: {record["sample_key"]: record for record in report["records"]}
        for backend, report in available.items()
    }
    consensus_root = experiment / "consensus"
    preview_root = experiment / "previews"
    consensus_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)
    sample_reports: list[dict[str, Any]] = []
    object_panels: dict[str, list[np.ndarray]] = defaultdict(list)

    for sample in manifest["samples"]:
        source = cv2.imread(str(root / sample["image_path"]), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(root / sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if source is None or mask_u8 is None:
            raise ValueError(f"Unreadable evidence for {sample['sample_key']}")
        mask = mask_u8 > 127
        normalized: dict[str, np.ndarray] = {}
        metric_summaries: dict[str, Any] = {}
        for backend, records in record_lookup.items():
            record = records.get(sample["sample_key"])
            if record is None:
                continue
            depth = np.load(root / record["artifacts"]["depth_m"]).astype(np.float32)
            if depth.shape != mask.shape or not np.isfinite(depth).all():
                raise ValueError(f"Invalid array for {sample['sample_key']}:{backend}")
            norm, p02, p98 = robust_normalize(depth, mask)
            normalized[backend] = norm
            metric_summaries[backend] = {
                "p02_m": p02,
                "median_m": float(np.median(depth[mask])),
                "p98_m": p98,
                "span_m": p98 - p02,
                "metric_status": "unvalidated_model_hypothesis",
            }

        names = sorted(normalized)
        pairs = []
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                correlation = masked_correlation(
                    normalized[left_name], normalized[right_name], mask
                )
                pairs.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "pearson_shape_correlation": correlation,
                        "median_absolute_normalized_difference": float(
                            np.median(
                                np.abs(normalized[left_name][mask] - normalized[right_name][mask])
                            )
                        ),
                    }
                )

        artifacts: dict[str, str] = {}
        panel_parts = [panel(source, f"{sample['view'].upper()} SOURCE")]
        for name in names:
            panel_parts.append(
                panel(
                    colorize_depth(normalized[name], mask),
                    name.replace("_", " ").upper(),
                    legend="depth",
                )
            )
        if len(normalized) >= 2:
            stack = np.stack([normalized[name] for name in names], axis=0)
            consensus = np.median(stack, axis=0).astype(np.float32)
            disagreement = np.median(np.abs(stack - consensus[None]), axis=0).astype(np.float32)
            disagreement /= max(float(np.percentile(disagreement[mask], 98)), 1e-6)
            disagreement = np.clip(disagreement, 0, 1)
            object_dir = consensus_root / sample["object_id"]
            object_dir.mkdir(parents=True, exist_ok=True)
            consensus_path = object_dir / f"{sample['view']}_normalized_consensus.npy"
            disagreement_path = object_dir / f"{sample['view']}_normalized_disagreement.npy"
            np.save(consensus_path, consensus)
            np.save(disagreement_path, disagreement)
            artifacts = {
                "normalized_consensus": str(consensus_path.relative_to(root)),
                "normalized_disagreement": str(disagreement_path.relative_to(root)),
                "consensus_sha256": sha256(consensus_path),
                "disagreement_sha256": sha256(disagreement_path),
            }
            panel_parts.extend(
                [
                    panel(
                        colorize_depth(consensus, mask),
                        "MEDIAN CONSENSUS",
                        legend="depth",
                    ),
                    panel(
                        colorize_magnitude(disagreement, mask),
                        "MODEL DISAGREEMENT",
                        legend="magnitude",
                    ),
                ]
            )
            stability = float(1.0 - np.mean(disagreement[mask]))
            status = "cross_model_stability_only"
        else:
            stability = None
            status = "insufficient_models_for_consensus"

        row = cv2.hconcat(panel_parts)
        object_panels[sample["object_id"]].append(row)
        sample_reports.append(
            {
                "sample_key": sample["sample_key"],
                "object_id": sample["object_id"],
                "view": sample["view"],
                "model_count": len(normalized),
                "models": names,
                "metric_depth_summaries": metric_summaries,
                "pairwise": pairs,
                "stability_score": stability,
                "status": status,
                "artifacts": artifacts,
            }
        )

    review_sheets: dict[str, str] = {}
    for object_id, rows in object_panels.items():
        width = max(row.shape[1] for row in rows)
        padded = [
            cv2.copyMakeBorder(
                row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(246, 246, 246)
            )
            for row in rows
        ]
        sheet = cv2.vconcat(padded)
        path = preview_root / f"{object_id}_metric_depth_consensus.png"
        if not cv2.imwrite(str(path), sheet):
            raise OSError(f"Failed to write {path}")
        review_sheets[object_id] = str(path.relative_to(root))

    stable_values = [
        record["stability_score"]
        for record in sample_reports
        if record["stability_score"] is not None
    ]
    completed = len(available)
    report = {
        "schema_version": "image2cad_metric_depth_consensus_v1",
        "created_at_utc": utc_now(),
        "stage": "image_only_phase2_metric_depth_consensus",
        "input_manifest": str((experiment / "input_manifest.json").relative_to(root)),
        "input_manifest_sha256": sha256(experiment / "input_manifest.json"),
        "available_backends": sorted(available),
        "blocked_backends": blocked,
        "incomplete_backends": incomplete,
        "backend_count": completed,
        "backend_coverage": {
            backend: {
                "sample_count": report["sample_count"],
                "requested_sample_count": manifest["sample_count"],
                "coverage_complete": report["sample_count"] == manifest["sample_count"],
            }
            for backend, report in available.items()
        },
        "sample_count": len(sample_reports),
        "samples": sample_reports,
        "review_sheets": review_sheets,
        "summary": {
            "consensus_ready": completed >= 2,
            "requested_four_model_ensemble_complete": completed == 4
            and all(report["sample_count"] == manifest["sample_count"] for report in available.values()),
            "mean_cross_model_stability": float(np.mean(stable_values)) if stable_values else None,
            "minimum_cross_model_stability": float(np.min(stable_values)) if stable_values else None,
        },
        "same_view_alignment": {
            "spatial_registration": "source_pixel_grid_after_resize_to_source_mask",
            "normalization": "per_model_masked_p02_p98",
            "aggregation": "pixelwise_median_normalized_depth",
            "uncertainty": "median_absolute_deviation_from_consensus",
            "complete": completed >= 2,
        },
        "visualization": {
            "depth_array_convention": "larger_metric_depth_is_farther",
            "depth_display_transform": "display_proximity_equals_one_minus_normalized_depth",
            "depth_color_legend": {
                "red": "near",
                "blue": "far",
            },
            "disagreement_color_legend": {
                "red": "high_disagreement",
                "blue": "low_disagreement",
            },
            "numeric_depth_or_consensus_modified": False,
        },
        "geometry_alignment": {
            "colmap_status": "failed_to_form_sparse_models_on_retained_STL1_run",
            "vggt_status": "blocked_checkpoint_and_4_gib_vram",
            "mast3r_status": "not_installed",
            "cross_view_metric_alignment_complete": False,
        },
        "claims": {
            "cross_model_stability_measured": completed >= 2,
            "depth_accuracy_measured": False,
            "absolute_scale_validated": False,
            "millimetre_measurements_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "decision": "research_only",
    }
    report_path = experiment / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1"),
    )
    parser.add_argument("--stl1-objects", nargs="*", default=list(DEFAULT_STL1_OBJECTS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = (root / args.output_dir).resolve()
    manifest_path = output / "input_manifest.json"
    if args.prepare_only or not manifest_path.exists():
        manifest = build_input_manifest(root, output, tuple(args.stl1_objects))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = manifest if args.prepare_only else aggregate(root, output, manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
