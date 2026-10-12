"""Build image-only surface-normal evidence from retained depth predictions.

The available images have no camera calibration or normal ground truth.  This
stage therefore preserves two distinct machine hypotheses:

* Metric3D's directly predicted camera-space normal field; and
* screen-space normal proxies derived from each retained metric-depth map.

Agreement is useful as uncertainty evidence, but is never reported as normal
accuracy or manufacturing geometry.
"""

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


COMPLETE_DEPTH_BACKENDS = (
    "depth_anything_v2_metric",
    "metric3d_v2",
    "unidepth_v2",
)


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
        return None


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(lengths, 1e-8)


def robust_normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    low, high = np.percentile(selected, (2, 98))
    if high <= low:
        high = low + 1e-6
    return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)


def depth_to_screen_normal(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert depth to a non-metric screen-space orientation proxy."""

    normalized = robust_normalize(np.asarray(depth, dtype=np.float32), mask)
    smoothed = cv2.GaussianBlur(normalized, (0, 0), 1.1)
    dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx * dx + dy * dy)
    scale = float(np.percentile(magnitude[mask], 95))
    scale = max(scale, 1e-6)
    normals = np.stack((-dx / scale, -dy / scale, np.ones_like(dx)), axis=-1)
    normals = normalize_vectors(normals)
    normals[~mask] = 0
    return normals.astype(np.float32)


def orient_direct_normals(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Orient direct normals toward positive camera Z for comparison/display."""

    normals = normalize_vectors(values)
    flipped = float(np.median(normals[..., 2][mask])) < 0
    if flipped:
        normals = -normals
    normals[~mask] = 0
    return normals.astype(np.float32), flipped


def angular_error_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dots = np.sum(normalize_vectors(left) * normalize_vectors(right), axis=-1)
    return np.degrees(np.arccos(np.clip(dots, -1, 1))).astype(np.float32)


def normal_visualization(normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.round(np.clip(normals * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[~mask] = (242, 242, 242)
    return bgr


def magnitude_visualization(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    high = float(np.percentile(values[mask], 98))
    normalized = np.clip(values / max(high, 1e-6), 0, 1)
    color = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~mask] = (242, 242, 242)
    return color


def panel(image: np.ndarray, title: str, width: int = 260) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((resized.shape[0] + 34, width, 3), 246, np.uint8)
    canvas[34:] = resized
    cv2.putText(canvas, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def build_review_sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    row_width = max(sum(image.shape[1] for image in row) for row in rows)
    assembled = []
    for row in rows:
        height = max(image.shape[0] for image in row)
        canvas = np.full((height, row_width, 3), 246, np.uint8)
        x = 0
        for image in row:
            canvas[: image.shape[0], x : x + image.shape[1]] = image
            x += image.shape[1]
        assembled.append(canvas)
    return np.vstack(assembled)


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path, depth_experiment: Path, output: Path, force: bool = False) -> dict[str, Any]:
    root = root.resolve()
    depth_experiment = depth_experiment.resolve()
    output = output.resolve()
    report_path = output / "report.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} exists; use --force to rebuild")

    depth_report = _read_report(depth_experiment / "report.json")
    manifest = _read_report(depth_experiment / "input_manifest.json")
    backend_reports = {
        backend: _read_report(depth_experiment / "backends" / backend / "backend_report.json")
        for backend in COMPLETE_DEPTH_BACKENDS
    }
    records_by_backend = {
        backend: {record["sample_key"]: record for record in report["records"]}
        for backend, report in backend_reports.items()
    }
    direct_records = records_by_backend["metric3d_v2"]
    output.mkdir(parents=True, exist_ok=True)
    previews_dir = output / "previews"
    previews_dir.mkdir(exist_ok=True)

    sample_results: list[dict[str, Any]] = []
    review_rows: dict[str, list[list[np.ndarray]]] = defaultdict(list)
    for sample in manifest["samples"]:
        sample_key = sample["sample_key"]
        object_id = sample["object_id"]
        view = sample["view"]
        source = cv2.imread(str(root / sample["image_path"]), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(root / sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if source is None or mask_u8 is None:
            raise ValueError(f"Cannot decode source evidence for {sample_key}")
        mask = mask_u8 > 127

        proxy_normals = []
        proxy_sources: dict[str, str] = {}
        for backend in COMPLETE_DEPTH_BACKENDS:
            record = records_by_backend[backend][sample_key]
            depth = np.load(root / record["artifacts"]["depth_m"])
            proxy = depth_to_screen_normal(depth, mask)
            proxy_normals.append(proxy)
            proxy_sources[backend] = record["artifacts"]["depth_m"]

        proxy_stack = np.stack(proxy_normals)
        consensus = normalize_vectors(np.mean(proxy_stack, axis=0))
        consensus[~mask] = 0
        proxy_angles = np.stack(
            [angular_error_degrees(proxy, consensus) for proxy in proxy_normals]
        )
        disagreement = np.mean(proxy_angles, axis=0).astype(np.float32)
        disagreement[~mask] = 0

        direct_record = direct_records[sample_key]
        direct_path = root / direct_record["artifacts"]["normals"]
        direct_raw = np.load(direct_path)
        direct, direct_flipped = orient_direct_normals(direct_raw, mask)
        direct_delta = angular_error_degrees(direct, consensus)
        direct_delta[~mask] = 0

        sample_results.append(
            {
                "sample_key": sample_key,
                "object_id": object_id,
                "split": sample.get("split"),
                "view": view,
                "status": "machine_normal_hypothesis_only",
                "source_sha256": sample["source_sha256"],
                "direct_normal": {
                    "backend": "metric3d_v2",
                    "source_path": direct_record["artifacts"]["normals"],
                    "sign_flipped_for_positive_camera_z": direct_flipped,
                    "source_file_sha256": sha256(direct_path),
                    "oriented_array_sha256": array_sha256(direct),
                },
                "depth_normal_proxy_sources": proxy_sources,
                "depth_normal_consensus_sha256": array_sha256(consensus),
                "normal_disagreement_sha256": array_sha256(disagreement),
                "metrics": {
                    "mean_depth_proxy_disagreement_deg": float(np.mean(disagreement[mask])),
                    "p95_depth_proxy_disagreement_deg": float(np.percentile(disagreement[mask], 95)),
                    "mean_direct_vs_depth_consensus_deg": float(np.mean(direct_delta[mask])),
                    "p95_direct_vs_depth_consensus_deg": float(np.percentile(direct_delta[mask], 95)),
                },
            }
        )
        review_rows[object_id].append(
            [
                panel(source, f"{view.upper()} SOURCE"),
                panel(normal_visualization(direct, mask), "METRIC3D DIRECT NORMAL"),
                panel(normal_visualization(consensus, mask), "DEPTH-NORMAL CONSENSUS"),
                panel(magnitude_visualization(disagreement, mask), "MODEL DISAGREEMENT"),
                panel(magnitude_visualization(direct_delta, mask), "DIRECT / DEPTH DELTA"),
            ]
        )

    review_sheets: dict[str, str] = {}
    for object_id, rows in review_rows.items():
        path = previews_dir / f"{object_id}_surface_normal_evidence.png"
        if not cv2.imwrite(str(path), build_review_sheet(rows)):
            raise OSError(f"Cannot write {path}")
        review_sheets[object_id] = path.relative_to(root).as_posix()

    mean_disagreement = [item["metrics"]["mean_depth_proxy_disagreement_deg"] for item in sample_results]
    mean_direct_delta = [item["metrics"]["mean_direct_vs_depth_consensus_deg"] for item in sample_results]
    report = {
        "schema_version": "image2cad_surface_normal_evidence_v1",
        "created_at_utc": utc_now(),
        "code_commit": git_commit(root),
        "stage": "camera_depth_normal_evidence_benchmark",
        "execution_profile": "images_only",
        "source_depth_report": (depth_experiment / "report.json").relative_to(root).as_posix(),
        "source_depth_report_sha256": sha256(depth_experiment / "report.json"),
        "sample_count": len(sample_results),
        "object_count": len(review_sheets),
        "backends": {
            "metric3d_direct_normal": {
                **BACKEND_MODELS["metric3d_v2"],
                "status": "available",
                "role": "direct_camera_space_normal_hypothesis",
            },
            "depth_gradient_proxies": {
                "status": "available",
                "backends": list(COMPLETE_DEPTH_BACKENDS),
                "role": "screen_space_orientation_and_disagreement",
            },
            "dsine": {
                "model_id": "baegwangbin/DSINE exp001_cvpr2024",
                "status": "not_run",
                "blockers": ["code_and_checkpoint_not_installed"],
                "role": "independent_surface_normal_comparator",
            },
        },
        "coordinate_contract": {
            "direct_normal_source": "metric3d_camera_space_model_output",
            "comparison_orientation": "positive_camera_z_after_recorded_global_sign_alignment",
            "depth_normal_proxy": "non_metric_screen_space_gradient_after_masked_p02_p98_depth_normalization",
            "cross_view_frame_alignment": False,
        },
        "summary": {
            "mean_depth_proxy_disagreement_deg": float(np.mean(mean_disagreement)),
            "mean_direct_vs_depth_consensus_deg": float(np.mean(mean_direct_delta)),
            "normal_ground_truth_available": False,
            "normal_accuracy_measured": False,
            "cross_view_normal_fusion_complete": False,
        },
        "visualization": {
            "normal_rgb": {"red": "+x", "green": "+y", "blue": "+z"},
            "disagreement_color_legend": {"blue": "low", "red": "high"},
        },
        "review_sheets": review_sheets,
        "samples": sample_results,
        "claims": {
            "per_pixel_normal_hypotheses_available": True,
            "multi_depth_normal_disagreement_measured": True,
            "normal_accuracy_measured": False,
            "metric_surface_orientation_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "No surface-normal ground truth is available for Ring01 or STL-1.",
            "Depth-derived normals are screen-space proxies because camera intrinsics and metric scale are unknown.",
            "Polished metal, gemstones, shadows and highlights can bias both depth and direct-normal models.",
            "DSINE remains an unexecuted independent comparator until pinned code and checkpoint evidence are installed.",
        ],
        "decision": "research_only_continue_to_independent_normal_and_cross_view_benchmark",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--depth-experiment",
        type=Path,
        default=Path("dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/phase_runs/v1/surface_normal_evidence_v1"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    depth_experiment = args.depth_experiment
    if not depth_experiment.is_absolute():
        depth_experiment = root / depth_experiment
    output = args.output
    if not output.is_absolute():
        output = root / output
    report = run(root, depth_experiment, output, args.force)
    print(json.dumps({"report": str(output / "report.json"), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
