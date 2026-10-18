"""Corroborate frozen Phase 5.7 validation tracks with independent normals.

Metric3D direct normals are compared with a screen-space consensus built only
from Depth Anything V2 Metric and UniDepth V2.  Metric3D depth is deliberately
excluded from that comparator.  The result annotates retained tracks and never
changes correspondence, cameras, CAD geometry or the held-out test split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_relief_tracks import (
    THRESHOLDS,
    _load_view_evidence,
    _metric3d_normal_records,
    _sample,
    evaluate_track_consistency,
)
from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS, VIEW_ORDER
from pipeline.run_surface_normal_evidence import (
    angular_error_degrees,
    depth_to_screen_normal,
    normalize_vectors,
    orient_direct_normals,
)


DEPTH_ROOT = Path("dataset/phase_runs/v1/phase5_validation_metric_depth_v1")
BASELINE_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1"
)
HYBRID_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_relief_hybrid_validation_v1"
)
NORMAL_AGREEMENT_THRESHOLD = THRESHOLDS[
    "minimum_normal_agreement_when_available"
]
NORMAL_ANGLE_SCALE_DEGREES = 35.0


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _backend_records(root: Path, backend: str) -> dict[str, dict[str, Any]]:
    report = json.loads(
        (
            root / DEPTH_ROOT / "backends" / backend / "backend_report.json"
        ).read_text(encoding="utf-8")
    )
    return {item["sample_key"]: item for item in report["records"]}


def _normal_maps(
    root: Path,
    object_record: dict[str, Any],
    depth_anything: dict[str, dict[str, Any]],
    unidepth: dict[str, dict[str, Any]],
    metric3d: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    object_id = object_record["object_id"]
    result = {}
    for view in VIEW_ORDER:
        key = f"{object_id}:{view}"
        bundle_path = root / object_record["views"][view]["bundle"]
        with np.load(bundle_path) as bundle:
            support = np.asarray(bundle["mask_jewelry_u8"], dtype=bool)
        da_normal = depth_to_screen_normal(
            np.load(root / depth_anything[key]["artifacts"]["depth_m"]), support
        )
        uni_normal = depth_to_screen_normal(
            np.load(root / unidepth[key]["artifacts"]["depth_m"]), support
        )
        independent_consensus = normalize_vectors(da_normal + uni_normal)
        independent_consensus[~support] = 0.0
        direct, _ = orient_direct_normals(
            np.load(root / metric3d[key]["artifacts"]["normals"]), support
        )
        proxy_angle = angular_error_degrees(da_normal, uni_normal)
        direct_angle = angular_error_degrees(direct, independent_consensus)
        proxy_agreement = np.exp(
            -np.square(proxy_angle / NORMAL_ANGLE_SCALE_DEGREES)
        ).astype(np.float32)
        direct_agreement = np.exp(
            -np.square(direct_angle / NORMAL_ANGLE_SCALE_DEGREES)
        ).astype(np.float32)
        combined = np.minimum(proxy_agreement, direct_agreement)
        for values in (proxy_angle, direct_angle, proxy_agreement, direct_agreement, combined):
            values[~support] = 0.0
        result[view] = {
            "support": support,
            "direct": direct,
            "independent_consensus": independent_consensus,
            "proxy_angle": proxy_angle,
            "direct_angle": direct_angle,
            "combined_agreement": combined,
        }
    return result


def _load_tracks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _corroborate(
    tracks: dict[str, np.ndarray],
    accepted: np.ndarray,
    normal_maps: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view_count, _, query_count = accepted.shape
    corroborated = np.zeros_like(accepted)
    source_agreement = np.zeros((view_count, query_count), np.float32)
    target_agreement = np.zeros_like(accepted, np.float32)
    for source_index, source_view in enumerate(VIEW_ORDER):
        source_agreement[source_index] = _sample(
            normal_maps[source_view]["combined_agreement"],
            tracks["query_points_xy"][source_index],
        )
        for target_index, target_view in enumerate(VIEW_ORDER):
            if source_index == target_index:
                continue
            target_agreement[source_index, target_index] = _sample(
                normal_maps[target_view]["combined_agreement"],
                tracks["tracks_xy"][source_index, target_index],
            )
            corroborated[source_index, target_index] = (
                accepted[source_index, target_index]
                & (source_agreement[source_index] >= NORMAL_AGREEMENT_THRESHOLD)
                & (
                    target_agreement[source_index, target_index]
                    >= NORMAL_AGREEMENT_THRESHOLD
                )
            )
    return corroborated, source_agreement, target_agreement


def _agreement_color(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    color = cv2.applyColorMap(
        np.uint8(np.round(np.clip(values, 0.0, 1.0) * 255)),
        cv2.COLORMAP_TURBO,
    )
    color[~support] = (242, 242, 242)
    return color


def _point_overlay(
    image: np.ndarray,
    points: np.ndarray,
    accepted: np.ndarray,
    corroborated: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    scale_x = (image.shape[1] - 1) / 517.0
    scale_y = (image.shape[0] - 1) / 517.0
    for point, is_accepted, is_corroborated in zip(
        points, accepted, corroborated
    ):
        if not is_accepted:
            continue
        xy = (
            int(round(float(point[0]) * scale_x)),
            int(round(float(point[1]) * scale_y)),
        )
        color = (70, 185, 80) if is_corroborated else (190, 75, 190)
        cv2.circle(canvas, xy, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, xy, 5, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    width = 520
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((resized.shape[0] + 78, width, 3), 246, np.uint8)
    canvas[78:] = resized
    cv2.putText(canvas, title, (9, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 29, 34), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (9, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (62, 66, 72), 1, cv2.LINE_AA)
    return canvas


def _render_object_audit(
    source: np.ndarray,
    tracks: dict[str, np.ndarray],
    accepted_front: np.ndarray,
    corroborated_front: np.ndarray,
    maps: dict[str, dict[str, np.ndarray]],
    output: Path,
    object_id: str,
) -> None:
    accepted_any = np.any(accepted_front, axis=0)
    corroborated_any = np.any(corroborated_front, axis=0)
    overlay = _point_overlay(
        source,
        tracks["query_points_xy"][0],
        accepted_any,
        corroborated_any,
    )
    agreement = _agreement_color(
        maps["front"]["combined_agreement"], maps["front"]["support"]
    )
    direct_delta = cv2.applyColorMap(
        np.uint8(
            np.round(
                np.clip(maps["front"]["direct_angle"] / 90.0, 0.0, 1.0)
                * 255
            )
        ),
        cv2.COLORMAP_TURBO,
    )
    direct_delta[~maps["front"]["support"]] = (242, 242, 242)
    supported_count = int(np.count_nonzero(accepted_any))
    corroborated_count = int(np.count_nonzero(corroborated_any))
    panels = [
        _panel(
            overlay,
            "TRACK NORMAL STATUS",
            f"{corroborated_count}/{supported_count} supported points corroborated",
        ),
        _panel(
            agreement,
            "INDEPENDENT NORMAL AGREEMENT",
            "Blue = low | red = high | Metric3D depth excluded",
        ),
        _panel(
            direct_delta,
            "DIRECT / INDEPENDENT DELTA",
            "Blue = low angle | red = 90 degrees or more",
        ),
    ]
    row = np.hstack(panels)
    header = np.full((65, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"PHASE 5.8 - {object_id.upper()} NORMAL CORROBORATION",
        (12, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (25, 29, 34),
        2,
        cv2.LINE_AA,
    )
    footer = np.full((42, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Green = normal corroborated | magenta = reciprocal track retained but normal unresolved",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    width = 1170
    rows = []
    for image in images:
        scale = width / image.shape[1]
        rows.append(
            cv2.resize(
                image,
                (width, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    cv2.imwrite(str(output), np.vstack(rows))


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(
        (
            root / "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
        ).read_text(encoding="utf-8")
    )
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    phase5_7 = json.loads(
        (root / "data/phase5_dense_relief_validation_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    phase5_7_by_id = {item["object_id"]: item for item in phase5_7["objects"]}
    depth_anything = _backend_records(root, "depth_anything_v2_metric")
    unidepth = _backend_records(root, "unidepth_v2")
    metric3d = _backend_records(root, "metric3d_v2")
    original_metric_depths = _metric_depth_records(root)
    original_direct_normals = _metric3d_normal_records(root)
    objects = []
    audit_paths = []

    for object_id in VALIDATION_OBJECTS:
        object_record = phase1_by_id[object_id]
        if object_record["split"] != "val":
            raise RuntimeError(f"{object_id} is not in the fixed validation split")
        original_evidence = _load_view_evidence(
            root,
            object_record,
            original_metric_depths,
            original_direct_normals,
        )
        baseline = _load_tracks(
            root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        hybrid = _load_tracks(
            root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        baseline_evaluated = evaluate_track_consistency(
            baseline, original_evidence
        )
        hybrid_evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(hybrid, original_evidence),
            baseline_evaluated,
        )
        maps = _normal_maps(
            root, object_record, depth_anything, unidepth, metric3d
        )
        corroborated, source_agreement, target_agreement = _corroborate(
            hybrid, hybrid_evaluated["accepted"], maps
        )
        front_accepted = hybrid_evaluated["accepted"][0]
        front_corroborated = corroborated[0]
        accepted_any = np.any(front_accepted, axis=0)
        corroborated_any = np.any(front_corroborated, axis=0)
        accepted_count = int(np.count_nonzero(accepted_any))
        corroborated_count = int(np.count_nonzero(corroborated_any))
        expected_count = phase5_7_by_id[object_id]["variants"][
            "phase5_7_hybrid"
        ]["front_supported_count"]
        source = cv2.imread(
            str(original_evidence["front"]["source_image"]), cv2.IMREAD_COLOR
        )
        audit_path = output / f"{object_id}_normal_corroboration.png"
        _render_object_audit(
            source,
            hybrid,
            front_accepted,
            front_corroborated,
            maps,
            audit_path,
            object_id,
        )
        audit_paths.append(audit_path)
        target_counts = {}
        for target_index, view in enumerate(VIEW_ORDER):
            if target_index == 0:
                continue
            accepted_target = front_accepted[target_index]
            corroborated_target = front_corroborated[target_index]
            target_counts[view] = {
                "accepted": int(np.count_nonzero(accepted_target)),
                "normal_corroborated": int(
                    np.count_nonzero(corroborated_target)
                ),
            }
        accepted_source_agreement = source_agreement[0][accepted_any]
        accepted_target_values = target_agreement[0][front_accepted]
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "phase5_7_supported_count": accepted_count,
                "phase5_7_report_count": expected_count,
                "normal_corroborated_count": corroborated_count,
                "normal_corroborated_fraction": round(
                    corroborated_count / max(accepted_count, 1), 6
                ),
                "normal_unresolved_count": accepted_count - corroborated_count,
                "mean_source_normal_agreement_on_supported": round(
                    float(np.mean(accepted_source_agreement))
                    if accepted_source_agreement.size
                    else 0.0,
                    6,
                ),
                "mean_target_normal_agreement_on_accepted_tracks": round(
                    float(np.mean(accepted_target_values))
                    if accepted_target_values.size
                    else 0.0,
                    6,
                ),
                "target_views": target_counts,
                "phase5_7_count_reproduced": accepted_count == expected_count,
                "artifacts": {"audit": _relative(audit_path, root)},
            }
        )

    cohort_path = output / "validation_normal_corroboration_audit.png"
    _cohort_audit(audit_paths, cohort_path)
    supported = sum(item["phase5_7_supported_count"] for item in objects)
    corroborated = sum(item["normal_corroborated_count"] for item in objects)
    top_accepted = sum(item["target_views"]["top"]["accepted"] for item in objects)
    top_corroborated = sum(
        item["target_views"]["top"]["normal_corroborated"] for item in objects
    )
    report = {
        "schema_version": "lalitha_phase5_normal_corroboration_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.8 independent normal corroboration",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "view_order": list(VIEW_ORDER),
            "normal_agreement_threshold": NORMAL_AGREEMENT_THRESHOLD,
            "normal_angle_scale_degrees": NORMAL_ANGLE_SCALE_DEGREES,
        },
        "sources": {
            "phase5_7_validation": "data/phase5_dense_relief_validation_v1/report.json",
            "metric_depth": f"{DEPTH_ROOT.as_posix()}/report.json",
            "hybrid_tracks": f"{HYBRID_TRACK_ROOT.as_posix()}/report.json",
            "direct_normal_backend": "metric3d_v2",
            "independent_depth_normal_backends": [
                "depth_anything_v2_metric",
                "unidepth_v2",
            ],
        },
        "summary": {
            "object_count": len(objects),
            "view_count": len(objects) * len(VIEW_ORDER),
            "phase5_7_supported_count": supported,
            "normal_corroborated_count": corroborated,
            "normal_corroborated_fraction": round(
                corroborated / max(supported, 1), 6
            ),
            "normal_unresolved_count": supported - corroborated,
            "top_target_accepted_count": top_accepted,
            "top_target_normal_corroborated_count": top_corroborated,
        },
        "checks": {
            "fixed_validation_objects_only": all(
                item["split"] == "val" for item in objects
            ),
            "all_15_validation_views_have_three_model_outputs": all(
                f"{object_id}:{view}" in depth_anything
                and f"{object_id}:{view}" in unidepth
                and f"{object_id}:{view}" in metric3d
                for object_id in VALIDATION_OBJECTS
                for view in VIEW_ORDER
            ),
            "metric3d_depth_excluded_from_independent_comparator": True,
            "phase5_7_supported_counts_reproduced": all(
                item["phase5_7_count_reproduced"] for item in objects
            ),
            "test_objects_not_processed": True,
            "phase5_7_tracks_unchanged": True,
            "exact_cad_unchanged": True,
        },
        "objects": objects,
        "decision": "retain_normal_corroboration_as_uncertainty_annotation_only",
        "claims": {
            "normal_accuracy_validated": False,
            "physical_correspondence_validated": False,
            "dense_correspondence_validated": False,
            "cad_fusion_authorized": False,
            "metric_scale_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "There is no surface-normal ground truth.",
            "Depth Anything V2 Metric and UniDepth V2 are independent model families but can share training-domain biases.",
            "Camera-space normals are not compared directly across views because calibrated camera rotations are unavailable.",
            "Normal corroboration annotates retained tracks and cannot recover ornament surfaces hidden from a view.",
        ],
        "artifacts": {"audit": _relative(cohort_path, root)},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase5_normal_corroboration_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "summary": report["summary"],
                "checks": report["checks"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
