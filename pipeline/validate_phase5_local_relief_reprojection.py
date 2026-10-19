"""Validate Phase 5.12 relief surfaces against independent target views.

The front-view OBJ trials are not refitted here.  Their existing vertices and
triangle connectivity are routed through retained reciprocal VGGT tracks into
normal-corroborated target views.  Source-derived relief values are then
compared with independently prepared target-view relief evidence.  Results are
review gates only: no STEP, CAD fusion, semantic label or physical-accuracy
claim is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_local_relief_surfaces import (
    NOMINAL_RELIEF_AMPLITUDE,
    _rasterize_triangles,
    _triangle_area,
    _track_to_image,
)
from pipeline.build_phase5_stl1_relief_evidence import (
    _metric_depth_records,
    build_relief_evidence,
)
from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS
from pipeline.subdivide_phase5_component_evidence import PHASE1_REPORT


LOCAL_SURFACE_ROOT = Path("data/phase5_local_relief_surfaces_v1")
SUBDIVISION_ROOT = Path("data/phase5_relief_subdivision_v1")
VISIBILITY_ROOT = Path("data/phase5_visibility_graph_v1")
TARGET_VIEWS = ("top", "iso", "lsv", "rsv")

MIN_TRACKED_QUERY_COUNT = 3
MIN_PROJECTED_FACE_COUNT = 1
MIN_PROJECTED_PIXEL_COUNT = 8
MIN_TARGET_SUPPORT_FRACTION = 0.25
MAX_TARGET_RELIEF_RMSE = 0.45
MIN_ZERO_BASELINE_IMPROVEMENT_FRACTION = 0.0
MIN_ORIENTATION_PRESERVATION_FRACTION = 0.75
MIN_TARGET_TRIANGLE_AREA_PIXELS2 = 1.0


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _normalize_relief(relief: np.ndarray, support: np.ndarray) -> np.ndarray:
    selected = np.abs(relief[support])
    scale = float(np.percentile(selected, 98.0)) if selected.size else 1.0
    return np.clip(relief / max(scale, 1e-8), -1.0, 1.0).astype(np.float32)


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(tuple(int(value) - 1 for value in line.split()[1:4]))
    return np.asarray(vertices, np.float64), np.asarray(faces, np.int32)


def _signed_triangle_area(points: np.ndarray) -> float:
    edge_a = points[1] - points[0]
    edge_b = points[2] - points[0]
    return float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]) * 0.5


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.round(points[:, 1]).astype(np.int32), 0, height - 1)
    return np.asarray(image[y, x], np.float32)


def _view_evidence(
    root: Path,
    object_id: str,
    view_record: dict[str, Any],
    view: str,
    metric_records: dict[str, tuple[Path, ...]],
) -> dict[str, np.ndarray]:
    with np.load(root / view_record["bundle"]) as bundle:
        evidence = build_relief_evidence(
            depth=bundle["relative_depth_regularized_f32"],
            uncertainty_u16=bundle["relative_depth_uncertainty_u16"],
            gradient=bundle["gradient_magnitude_f32"],
            local_contrast=bundle["local_contrast_f32"],
            jewelry_mask=bundle["mask_jewelry_u8"],
            opening_mask=bundle["mask_negative_space_proposal_u8"],
            metric_depths=tuple(
                np.load(path)
                for path in metric_records.get(f"{object_id}:{view}", ())
            ),
        )
        source = cv2.cvtColor(
            np.asarray(bundle["normalized_rgb_u8"]), cv2.COLOR_RGB2BGR
        )
    relief = np.asarray(evidence["relief"], np.float32)
    support = np.asarray(evidence["support"], dtype=bool)
    return {
        "source": source,
        "support": support,
        "relief": _normalize_relief(relief, support),
    }


def _evaluate_target(
    *,
    trial: dict[str, Any],
    query_node_ids: list[str],
    target_view: str,
    edge_by_query_view: dict[tuple[str, str], dict[str, Any]],
    target_evidence: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_points = np.asarray(trial["source_points_xy_image"], np.float32)
    obj_path = Path(trial["_absolute_obj_path"])
    vertices, source_faces = _load_obj(obj_path)
    if len(vertices) != len(source_points):
        raise RuntimeError(
            f"{trial['patch_id']} OBJ vertex count does not match query count"
        )
    source_relief = np.asarray(
        vertices[:, 2] / NOMINAL_RELIEF_AMPLITUDE, np.float32
    )
    height, width = target_evidence["relief"].shape
    target_points = np.full((len(query_node_ids), 2), np.nan, np.float32)
    target_edges: list[dict[str, Any] | None] = [None] * len(query_node_ids)
    for index, query_id in enumerate(query_node_ids):
        edge = edge_by_query_view.get((query_id, target_view))
        if edge is None or not edge["normal_corroborated"]:
            continue
        target_points[index] = _track_to_image(
            edge["target_xy"], width, height
        )
        target_edges[index] = edge

    tracked_indices = np.flatnonzero(np.all(np.isfinite(target_points), axis=1))
    retained_faces: list[np.ndarray] = []
    orientation_matches: list[bool] = []
    for face in source_faces:
        if not np.all(np.isfinite(target_points[face])):
            continue
        target_triangle = target_points[face]
        if _triangle_area(target_triangle) < MIN_TARGET_TRIANGLE_AREA_PIXELS2:
            continue
        source_area = _signed_triangle_area(source_points[face])
        target_area = _signed_triangle_area(target_triangle)
        retained_faces.append(face)
        orientation_matches.append(source_area * target_area > 0.0)
    projected_faces = (
        np.asarray(retained_faces, np.int32)
        if retained_faces
        else np.empty((0, 3), np.int32)
    )

    rendered, rendered_mask = _rasterize_triangles(
        target_points,
        source_relief,
        projected_faces,
        (height, width),
    )
    target_support = np.asarray(target_evidence["support"], dtype=bool)
    target_relief = np.asarray(target_evidence["relief"], np.float32)
    valid = rendered_mask & target_support
    projected_pixel_count = int(np.count_nonzero(rendered_mask))
    valid_pixel_count = int(np.count_nonzero(valid))
    support_fraction = valid_pixel_count / max(projected_pixel_count, 1)
    if valid_pixel_count:
        error = rendered[valid] - target_relief[valid]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        mae = float(np.mean(np.abs(error)))
        zero_rmse = float(np.sqrt(np.mean(np.square(target_relief[valid]))))
        improvement = (zero_rmse - rmse) / max(zero_rmse, 1e-8)
    else:
        rmse = None
        mae = None
        zero_rmse = None
        improvement = None

    tracked_points = target_points[tracked_indices]
    tracked_source_relief = source_relief[tracked_indices]
    tracked_target_relief = (
        _sample(target_relief, tracked_points)
        if len(tracked_points)
        else np.empty(0, np.float32)
    )
    tracked_support = (
        _sample(target_support.astype(np.uint8), tracked_points) > 0
        if len(tracked_points)
        else np.empty(0, bool)
    )
    informative = (
        tracked_support
        & (np.abs(tracked_source_relief) >= 0.05)
        & (np.abs(tracked_target_relief) >= 0.05)
    )
    sign_agreement = (
        float(
            np.mean(
                np.sign(tracked_source_relief[informative])
                == np.sign(tracked_target_relief[informative])
            )
        )
        if np.count_nonzero(informative)
        else None
    )
    edges = [target_edges[index] for index in tracked_indices]
    checks = {
        "minimum_tracked_queries": len(tracked_indices)
        >= MIN_TRACKED_QUERY_COUNT,
        "minimum_projected_faces": len(projected_faces)
        >= MIN_PROJECTED_FACE_COUNT,
        "minimum_projected_pixels": projected_pixel_count
        >= MIN_PROJECTED_PIXEL_COUNT,
        "minimum_target_support_fraction": support_fraction
        >= MIN_TARGET_SUPPORT_FRACTION,
        "maximum_target_relief_rmse": rmse is not None
        and rmse <= MAX_TARGET_RELIEF_RMSE,
        "beats_zero_relief_baseline": improvement is not None
        and improvement >= MIN_ZERO_BASELINE_IMPROVEMENT_FRACTION,
        "orientation_preserved": bool(orientation_matches)
        and float(np.mean(orientation_matches))
        >= MIN_ORIENTATION_PRESERVATION_FRACTION,
    }
    passed = all(checks.values())
    record = {
        "target_view": target_view,
        "tracked_query_count": int(len(tracked_indices)),
        "projected_face_count": int(len(projected_faces)),
        "projected_pixel_count": projected_pixel_count,
        "target_supported_pixel_count": valid_pixel_count,
        "target_support_fraction": round(float(support_fraction), 6),
        "target_relief_mae": round(mae, 6) if mae is not None else None,
        "target_relief_rmse": round(rmse, 6) if rmse is not None else None,
        "zero_relief_baseline_rmse": (
            round(zero_rmse, 6) if zero_rmse is not None else None
        ),
        "zero_baseline_improvement_fraction": (
            round(improvement, 6) if improvement is not None else None
        ),
        "orientation_preservation_fraction": round(
            float(np.mean(orientation_matches)) if orientation_matches else 0.0,
            6,
        ),
        "informative_sign_sample_count": int(np.count_nonzero(informative)),
        "relief_sign_agreement": (
            round(sign_agreement, 6) if sign_agreement is not None else None
        ),
        "mean_track_score": round(
            float(np.mean([edge["track_score"] for edge in edges])), 6
        )
        if edges
        else None,
        "mean_cycle_error_pixels": round(
            float(np.mean([edge["cycle_error_pixels"] for edge in edges])), 6
        )
        if edges
        else None,
        "checks": checks,
        "independent_target_gate_passed": passed,
    }
    visual = {
        "target_points": target_points,
        "projected_faces": projected_faces,
        "passed": passed,
    }
    return record, visual


def _render_panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    panel = cv2.resize(image, (420, 420), interpolation=cv2.INTER_AREA)
    canvas = np.full((484, 420, 3), 246, np.uint8)
    canvas[64:] = panel
    cv2.putText(
        canvas, title, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
        (25, 29, 34), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, subtitle, (9, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
        (60, 64, 69), 1, cv2.LINE_AA,
    )
    return canvas


def _summary_panel(object_report: dict[str, Any]) -> np.ndarray:
    canvas = np.full((484, 420, 3), 246, np.uint8)
    lines = [
        ("PHASE 5.13 SUMMARY", (25, 29, 34), 0.54),
        (f"Eligible patches: {object_report['eligible_patch_count']}", (55, 60, 65), 0.44),
        (f"Target evaluations: {object_report['target_evaluation_count']}", (55, 60, 65), 0.44),
        (f"Passed target gates: {object_report['passed_target_evaluation_count']}", (48, 126, 74), 0.44),
        (f"Single-target support: {object_report['single_target_supported_patch_count']}", (55, 60, 65), 0.44),
        (f"Multi-target support: {object_report['multi_target_supported_patch_count']}", (55, 60, 65), 0.44),
        (f"Rejected patches: {object_report['target_inconsistent_patch_count']}", (55, 60, 165), 0.44),
        (f"Unresolved patches: {object_report['unresolved_patch_count']}", (55, 60, 165), 0.44),
        ("Green = target gate passed", (48, 126, 74), 0.40),
        ("Red = failed or insufficient", (55, 60, 165), 0.40),
        ("No CAD fusion authorized", (55, 60, 165), 0.45),
    ]
    y = 34
    for text, color, scale in lines:
        cv2.putText(
            canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
            color, 1 if y > 34 else 2, cv2.LINE_AA,
        )
        y += 38
    return canvas


def _render_object_audit(
    *,
    object_id: str,
    view_evidence: dict[str, dict[str, np.ndarray]],
    patches: list[dict[str, Any]],
    visuals: dict[tuple[str, str], dict[str, Any]],
    object_report: dict[str, Any],
    output: Path,
) -> None:
    front = view_evidence["front"]["source"].copy()
    for patch in patches:
        points = np.asarray(patch["source_points_xy_image"], np.float32)
        if len(points) >= 3:
            hull = cv2.convexHull(np.round(points).astype(np.int32).reshape(-1, 1, 2))
            cv2.polylines(front, [hull], True, (190, 75, 155), 2, cv2.LINE_AA)
    panels = [
        _render_panel(
            front,
            "FRONT SURFACE SOURCE",
            f"{len(patches)} unchanged Phase 5.12 patch meshes",
        )
    ]
    for view in TARGET_VIEWS:
        overlay = view_evidence[view]["source"].copy()
        pass_count = 0
        evaluation_count = 0
        for patch in patches:
            visual = visuals.get((patch["patch_id"], view))
            if visual is None:
                continue
            evaluation_count += 1
            passed = visual["passed"]
            pass_count += int(passed)
            color = (70, 180, 85) if passed else (65, 65, 210)
            points = visual["target_points"]
            for face in visual["projected_faces"]:
                polygon = np.round(points[face]).astype(np.int32)
                layer = overlay.copy()
                cv2.fillConvexPoly(layer, polygon, color, cv2.LINE_AA)
                overlay = cv2.addWeighted(layer, 0.24, overlay, 0.76, 0.0)
                cv2.polylines(overlay, [polygon], True, color, 1, cv2.LINE_AA)
        panels.append(
            _render_panel(
                overlay,
                f"{view.upper()} TARGET EVIDENCE",
                f"{pass_count}/{evaluation_count} independent gates passed",
            )
        )
    panels.append(_summary_panel(object_report))
    grid = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    header = np.full((72, grid.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"PHASE 5.13 - {object_id.upper()} INDEPENDENT TARGET-VIEW REPROJECTION",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (25, 29, 34),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Front OBJ connectivity is projected through retained normal-corroborated tracks; no target refitting",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    footer = np.full((48, grid.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Image evidence only - not physical truth, metric scale, semantic validation or CAD authorization",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, grid, footer)))


def _cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    width = 1260
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
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    phase5_12 = json.loads(
        (root / LOCAL_SURFACE_ROOT / "report.json").read_text(encoding="utf-8")
    )
    phase5_12_by_id = {item["object_id"]: item for item in phase5_12["objects"]}
    metric_records = _metric_depth_records(root)
    objects = []
    audit_paths = []

    for object_id in VALIDATION_OBJECTS:
        source_record = phase1_by_id[object_id]
        if source_record["split"] != "val":
            raise RuntimeError(f"{object_id} is outside the fixed validation split")
        subdivision = json.loads(
            (root / SUBDIVISION_ROOT / object_id / "relief_subdivision.json").read_text(
                encoding="utf-8"
            )
        )
        eligible = [
            patch
            for patch in subdivision["patches"]
            if patch["future_surface_fit_review_eligible"]
        ]
        local_report = json.loads(
            (root / LOCAL_SURFACE_ROOT / object_id / "report.json").read_text(
                encoding="utf-8"
            )
        )
        trial_by_id = {trial["patch_id"]: trial for trial in local_report["trials"]}
        visibility = json.loads(
            (root / VISIBILITY_ROOT / object_id / "visibility_graph.json").read_text(
                encoding="utf-8"
            )
        )
        edge_by_query_view = {
            (edge["source_node_id"], edge["target_view"]): edge
            for edge in visibility["observation_edges"]
            if edge["reciprocal_accepted"]
        }
        evidence_by_view = {
            view: _view_evidence(
                root,
                object_id,
                source_record["views"][view],
                view,
                metric_records,
            )
            for view in ("front", *TARGET_VIEWS)
        }
        patch_reports = []
        patch_visuals: dict[tuple[str, str], dict[str, Any]] = {}
        audit_patches = []
        for patch in eligible:
            trial = dict(trial_by_id[patch["patch_id"]])
            trial["_absolute_obj_path"] = str(
                root / trial["artifact"]["obj"]
            )
            target_records = []
            for target_view in patch["normal_corroborated_target_views"]:
                record, visual = _evaluate_target(
                    trial=trial,
                    query_node_ids=patch["query_node_ids"],
                    target_view=target_view,
                    edge_by_query_view=edge_by_query_view,
                    target_evidence=evidence_by_view[target_view],
                )
                target_records.append(record)
                patch_visuals[(patch["patch_id"], target_view)] = visual
            pass_count = sum(
                record["independent_target_gate_passed"]
                for record in target_records
            )
            evaluable_count = sum(
                record["projected_face_count"] > 0 for record in target_records
            )
            if pass_count >= 2:
                state = "multi_target_supported_review"
            elif pass_count == 1:
                state = "single_target_supported_review"
            elif evaluable_count:
                state = "target_inconsistent_reject"
            else:
                state = "insufficient_projection_unresolved"
            patch_reports.append(
                {
                    "patch_id": patch["patch_id"],
                    "parent_component_id": patch["parent_component_id"],
                    "query_count": patch["query_count"],
                    "target_evaluation_count": len(target_records),
                    "passed_target_evaluation_count": pass_count,
                    "independent_target_state": state,
                    "target_evaluations": target_records,
                    "semantic_label": "unknown_ornamental_relief_patch",
                    "surface_promotion_authorized": False,
                    "cad_fusion_authorized": False,
                }
            )
            audit_patches.append(
                {
                    "patch_id": patch["patch_id"],
                    "source_points_xy_image": trial["source_points_xy_image"],
                }
            )

        target_evaluation_count = sum(
            item["target_evaluation_count"] for item in patch_reports
        )
        passed_target_evaluation_count = sum(
            item["passed_target_evaluation_count"] for item in patch_reports
        )
        state_counts = {
            state: sum(
                item["independent_target_state"] == state
                for item in patch_reports
            )
            for state in (
                "multi_target_supported_review",
                "single_target_supported_review",
                "target_inconsistent_reject",
                "insufficient_projection_unresolved",
            )
        }
        object_report = {
            "schema_version": "lalitha_phase5_target_reprojection_object_v1",
            "object_id": object_id,
            "split": "val",
            "eligible_patch_count": len(eligible),
            "target_evaluation_count": target_evaluation_count,
            "passed_target_evaluation_count": passed_target_evaluation_count,
            "multi_target_supported_patch_count": state_counts[
                "multi_target_supported_review"
            ],
            "single_target_supported_patch_count": state_counts[
                "single_target_supported_review"
            ],
            "target_inconsistent_patch_count": state_counts[
                "target_inconsistent_reject"
            ],
            "unresolved_patch_count": state_counts[
                "insufficient_projection_unresolved"
            ],
            "patches": patch_reports,
            "claims": {
                "independent_target_image_evidence_evaluated": True,
                "physical_relief_accuracy_validated": False,
                "semantic_accuracy_validated": False,
                "surface_promotion_authorized": False,
                "cad_fusion_authorized": False,
            },
        }
        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        object_report_path = object_dir / "report.json"
        audit_path = object_dir / "target_reprojection_audit.png"
        object_report["artifacts"] = {
            "report": _relative(object_report_path, root),
            "audit": _relative(audit_path, root),
        }
        object_report_path.write_text(
            json.dumps(object_report, indent=2) + "\n", encoding="utf-8"
        )
        _render_object_audit(
            object_id=object_id,
            view_evidence=evidence_by_view,
            patches=audit_patches,
            visuals=patch_visuals,
            object_report=object_report,
            output=audit_path,
        )
        audit_paths.append(audit_path)
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "eligible_patch_count": len(eligible),
                "phase5_12_eligible_count_reproduced": len(eligible)
                == phase5_12_by_id[object_id]["eligible_patch_count"],
                "target_evaluation_count": target_evaluation_count,
                "passed_target_evaluation_count": passed_target_evaluation_count,
                **state_counts,
                "artifacts": object_report["artifacts"],
            }
        )

    cohort_audit = output / "validation_target_reprojection_audit.png"
    _cohort_audit(audit_paths, cohort_audit)
    summary = {
        "object_count": len(objects),
        "eligible_patch_count": sum(item["eligible_patch_count"] for item in objects),
        "target_evaluation_count": sum(item["target_evaluation_count"] for item in objects),
        "passed_target_evaluation_count": sum(
            item["passed_target_evaluation_count"] for item in objects
        ),
        "multi_target_supported_patch_count": sum(
            item["multi_target_supported_review"] for item in objects
        ),
        "single_target_supported_patch_count": sum(
            item["single_target_supported_review"] for item in objects
        ),
        "target_inconsistent_patch_count": sum(
            item["target_inconsistent_reject"] for item in objects
        ),
        "unresolved_patch_count": sum(
            item["insufficient_projection_unresolved"] for item in objects
        ),
    }
    report = {
        "schema_version": "lalitha_phase5_target_reprojection_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.13 independent target-view reprojection validation",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "source_view": "front",
            "target_views": list(TARGET_VIEWS),
        },
        "sources": {
            "local_surfaces": f"{LOCAL_SURFACE_ROOT.as_posix()}/report.json",
            "subdivision": f"{SUBDIVISION_ROOT.as_posix()}/report.json",
            "visibility": f"{VISIBILITY_ROOT.as_posix()}/report.json",
            "phase1_features": PHASE1_REPORT.as_posix(),
        },
        "thresholds": {
            "minimum_tracked_query_count": MIN_TRACKED_QUERY_COUNT,
            "minimum_projected_face_count": MIN_PROJECTED_FACE_COUNT,
            "minimum_projected_pixel_count": MIN_PROJECTED_PIXEL_COUNT,
            "minimum_target_support_fraction": MIN_TARGET_SUPPORT_FRACTION,
            "maximum_target_relief_rmse": MAX_TARGET_RELIEF_RMSE,
            "minimum_zero_baseline_improvement_fraction": MIN_ZERO_BASELINE_IMPROVEMENT_FRACTION,
            "minimum_orientation_preservation_fraction": MIN_ORIENTATION_PRESERVATION_FRACTION,
            "minimum_target_triangle_area_pixels2": MIN_TARGET_TRIANGLE_AREA_PIXELS2,
        },
        "summary": summary,
        "checks": {
            "fixed_validation_objects_only": all(item["split"] == "val" for item in objects),
            "phase5_12_eligible_counts_reproduced": all(
                item["phase5_12_eligible_count_reproduced"] for item in objects
            ),
            "normal_corroborated_target_edges_only": True,
            "target_view_relief_not_used_to_refit_source_surfaces": True,
            "test_objects_not_processed": True,
            "exact_cad_unchanged": True,
            "no_step_files_generated": True,
        },
        "objects": objects,
        "decision": "retain_only_as_independent_target_view_review_evidence",
        "claims": {
            "independent_target_image_evidence_evaluated": True,
            "physical_relief_accuracy_validated": False,
            "semantic_accuracy_validated": False,
            "metric_scale_validated": False,
            "hidden_geometry_recovered": False,
            "surface_promotion_authorized": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Target coordinates come from retained reciprocal VGGT tracks and are not calibrated physical projections.",
            "Target relief is independent by view but is produced by the same conservative image-depth evidence family.",
            "Sparse projected triangles evaluate only observed front-surface patches and do not recover hidden geometry.",
            "A passed target gate supports review only; it does not establish semantic identity, metric scale or CAD accuracy.",
        ],
        "artifacts": {"audit": _relative(cohort_audit, root)},
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
        default=Path("data/phase5_target_reprojection_v1"),
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
