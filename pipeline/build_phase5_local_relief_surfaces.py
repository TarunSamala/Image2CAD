"""Build reversible local relief-surface trials for Phase 5.12.

Only Phase 5.11 patches already marked eligible for future fitting review are
triangulated.  Vertex height comes from the conservative Phase 5.4 signed
relief evidence, not raw image brightness.  Every OBJ remains an open,
scale-normalized evidence surface and is never fused into STEP or exact CAD.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

from pipeline.build_phase5_stl1_relief_evidence import (
    _metric_depth_records,
    _render_obj_preview,
    build_relief_evidence,
)
from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS
from pipeline.subdivide_phase5_component_evidence import (
    PHASE1_REPORT,
    TRACK_COORDINATE_MAX,
)


SUBDIVISION_ROOT = Path("data/phase5_relief_subdivision_v1")
NOMINAL_XY_WIDTH = 1.0
NOMINAL_RELIEF_AMPLITUDE = 0.025
MIN_TRIANGLE_AREA_PIXELS2 = 1.0
PALETTE_BGR = (
    (66, 174, 68),
    (190, 145, 35),
    (184, 70, 184),
    (55, 152, 210),
    (126, 105, 196),
    (192, 104, 42),
    (85, 180, 190),
    (165, 115, 70),
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _track_to_image(
    point_xy: list[float], width: int, height: int
) -> tuple[float, float]:
    return (
        float(point_xy[0]) * (width - 1) / TRACK_COORDINATE_MAX,
        float(point_xy[1]) * (height - 1) / TRACK_COORDINATE_MAX,
    )


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.round(points[:, 1]).astype(np.int32), 0, height - 1)
    return np.asarray(image[y, x], np.float32)


def _triangle_area(points: np.ndarray) -> float:
    edge_a = points[1] - points[0]
    edge_b = points[2] - points[0]
    return abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])) * 0.5


def _triangulate(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 3), np.int32)
    rounded = np.round(points, 4)
    if len(np.unique(rounded, axis=0)) != len(points):
        return np.empty((0, 3), np.int32)
    try:
        faces = np.asarray(Delaunay(points).simplices, np.int32)
    except QhullError:
        return np.empty((0, 3), np.int32)
    retained = []
    for face in faces:
        triangle = points[face]
        area = _triangle_area(triangle)
        if area >= MIN_TRIANGLE_AREA_PIXELS2:
            retained.append(face)
    return (
        np.asarray(retained, np.int32)
        if retained
        else np.empty((0, 3), np.int32)
    )


def _rasterize_triangles(
    points: np.ndarray,
    values: np.ndarray,
    faces: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    rendered = np.full((height, width), np.nan, np.float32)
    for face in faces:
        triangle = points[face]
        triangle_values = values[face]
        x0 = max(int(np.floor(triangle[:, 0].min())), 0)
        x1 = min(int(np.ceil(triangle[:, 0].max())), width - 1)
        y0 = max(int(np.floor(triangle[:, 1].min())), 0)
        y1 = min(int(np.ceil(triangle[:, 1].max())), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(x0, x1 + 1, dtype=np.float32),
            np.arange(y0, y1 + 1, dtype=np.float32),
        )
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (
            c[0] - b[0]
        ) * (a[1] - c[1])
        if abs(float(denominator)) < 1e-8:
            continue
        wa = (
            (b[1] - c[1]) * (grid_x - c[0])
            + (c[0] - b[0]) * (grid_y - c[1])
        ) / denominator
        wb = (
            (c[1] - a[1]) * (grid_x - c[0])
            + (a[0] - c[0]) * (grid_y - c[1])
        ) / denominator
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-5) & (wb >= -1e-5) & (wc >= -1e-5)
        values_grid = (
            wa * triangle_values[0]
            + wb * triangle_values[1]
            + wc * triangle_values[2]
        )
        target = rendered[y0 : y1 + 1, x0 : x1 + 1]
        target[inside] = values_grid[inside]
    return rendered, np.isfinite(rendered)


def _mesh_vertices(
    points_image: np.ndarray,
    normalized_relief: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    x = (points_image[:, 0] / max(width - 1, 1) - 0.5) * NOMINAL_XY_WIDTH
    y = (0.5 - points_image[:, 1] / max(height - 1, 1)) * NOMINAL_XY_WIDTH
    z = normalized_relief * NOMINAL_RELIEF_AMPLITUDE
    return np.column_stack((x, y, z)).astype(np.float64)


def _write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_id: str,
) -> None:
    lines = [
        "# Lalitha Phase 5.12 local reversible relief-surface trial",
        "# Nominal display units; not millimetres.",
        "# Open evidence surface; not a solid, STEP feature or manufacturing model.",
        f"g {patch_id}",
    ]
    lines.extend(
        f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices
    )
    lines.extend(
        f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}"
        for a, b, c in faces
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_combined_obj(
    path: Path,
    meshes: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    lines = [
        "# Lalitha Phase 5.12 combined local relief-surface trials",
        "# Groups remain independent, open and reversible.",
        "# Nominal display units; no CAD or manufacturing authorization.",
    ]
    offset = 0
    for patch_id, vertices, faces in meshes:
        lines.append(f"g {patch_id}")
        lines.extend(
            f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices
        )
        lines.extend(
            f"f {int(a) + 1 + offset} {int(b) + 1 + offset} {int(c) + 1 + offset}"
            for a, b, c in faces
        )
        offset += len(vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    width = 500
    height = 500
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 72, width, 3), 246, np.uint8)
    canvas[72:] = resized
    cv2.putText(canvas, title, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (25, 29, 34), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (9, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (65, 68, 73), 1, cv2.LINE_AA)
    return canvas


def _render_object_audit(
    source: np.ndarray,
    trials: list[dict[str, Any]],
    combined_preview_path: Path,
    output: Path,
    object_id: str,
) -> None:
    patch_view = source.copy()
    projection_view = source.copy()
    residual = np.full(source.shape[:2], np.nan, np.float32)
    for index, trial in enumerate(trials):
        color = PALETTE_BGR[index % len(PALETTE_BGR)]
        points = np.asarray(trial["source_points_xy_image"], np.float32)
        if len(points) >= 3:
            hull = cv2.convexHull(np.round(points).astype(np.int32).reshape(-1, 1, 2))
            cv2.polylines(patch_view, [hull], True, color, 2, cv2.LINE_AA)
        for point in np.round(points).astype(np.int32):
            cv2.circle(patch_view, tuple(point), 3, color, -1, cv2.LINE_AA)
        for face in trial["faces"]:
            polygon = np.round(points[np.asarray(face, np.int32)]).astype(np.int32)
            overlay = projection_view.copy()
            cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
            projection_view = cv2.addWeighted(
                overlay, 0.22, projection_view, 0.78, 0.0
            )
            cv2.polylines(projection_view, [polygon], True, color, 1, cv2.LINE_AA)
        error_path = trial.get("error_map_array")
        if error_path is not None:
            error_values = np.asarray(error_path, np.float32)
            mask = np.isfinite(error_values)
            residual[mask] = error_values[mask]

    residual_color = cv2.applyColorMap(
        np.uint8(np.round(np.clip(np.nan_to_num(residual) / 0.5, 0.0, 1.0) * 255)),
        cv2.COLORMAP_TURBO,
    )
    residual_color[~np.isfinite(residual)] = (235, 235, 235)
    panels = [
        _panel(
            patch_view,
            "ELIGIBLE PATCH QUERIES",
            f"{len(trials)} Phase 5.11 patches attempted",
        ),
        _panel(
            projection_view,
            "TRIANGULATED OBJ PROJECTION",
            "Independent groups; no STEP fusion",
        ),
        _panel(
            residual_color,
            "DENSE RELIEF INTERPOLATION ERROR",
            "Blue = low | red = 0.5 normalized error",
        ),
    ]
    row = np.hstack(panels)
    preview = cv2.imread(str(combined_preview_path), cv2.IMREAD_COLOR)
    target_width = row.shape[1]
    scale = target_width / preview.shape[1]
    preview = cv2.resize(
        preview,
        (target_width, int(round(preview.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    header = np.full((74, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"PHASE 5.12 - {object_id.upper()} LOCAL RELIEF-SURFACE TRIALS",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 29, 34),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Sparse patch triangulation from conservative signed-relief evidence",
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    footer = np.full((50, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Metrics compare against the same retained depth evidence; they do not measure physical or semantic accuracy.",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, preview, footer)))


def _cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    target_width = 1200
    rows = []
    for image in images:
        scale = target_width / image.shape[1]
        rows.append(
            cv2.resize(
                image,
                (target_width, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    cv2.imwrite(str(output), np.vstack(rows))


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subdivision_report = json.loads(
        (root / SUBDIVISION_ROOT / "report.json").read_text(encoding="utf-8")
    )
    subdivision_by_id = {
        item["object_id"]: item for item in subdivision_report["objects"]
    }
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
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
            item
            for item in subdivision["patches"]
            if item["future_surface_fit_review_eligible"]
        ]
        bundle_path = root / source_record["views"]["front"]["bundle"]
        with np.load(bundle_path) as bundle:
            source = cv2.cvtColor(
                np.asarray(bundle["normalized_rgb_u8"]), cv2.COLOR_RGB2BGR
            )
            metric_depths = tuple(
                np.load(path)
                for path in metric_records.get(f"{object_id}:front", ())
            )
            evidence = build_relief_evidence(
                depth=bundle["relative_depth_regularized_f32"],
                uncertainty_u16=bundle["relative_depth_uncertainty_u16"],
                gradient=bundle["gradient_magnitude_f32"],
                local_contrast=bundle["local_contrast_f32"],
                jewelry_mask=bundle["mask_jewelry_u8"],
                opening_mask=bundle["mask_negative_space_proposal_u8"],
                metric_depths=metric_depths,
            )
        relief = np.asarray(evidence["relief"], np.float32)
        support = np.asarray(evidence["support"], dtype=bool)
        confidence = np.asarray(evidence["confidence"], np.float32)
        selected = np.abs(relief[support])
        relief_scale = (
            float(np.percentile(selected, 98.0)) if selected.size else 1.0
        )
        normalized_relief_map = np.clip(
            relief / max(relief_scale, 1e-8), -1.0, 1.0
        )
        height, width = relief.shape
        object_dir = output / object_id
        patch_dir = object_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        trials = []
        combined_meshes = []

        for patch in eligible:
            points_image = np.asarray(
                [
                    _track_to_image(point, width, height)
                    for point in patch["source_points_xy"]
                ],
                np.float32,
            )
            faces = _triangulate(points_image)
            sampled_relief = _sample(normalized_relief_map, points_image)
            vertices = _mesh_vertices(
                points_image, sampled_relief, width, height
            )
            rendered, rendered_mask = _rasterize_triangles(
                points_image,
                sampled_relief,
                faces,
                (height, width),
            )
            valid_pixels = rendered_mask & support
            error = np.full((height, width), np.nan, np.float32)
            error[valid_pixels] = np.abs(
                rendered[valid_pixels] - normalized_relief_map[valid_pixels]
            )
            obj_path = patch_dir / f"{patch['patch_id']}.obj"
            mesh_generated = bool(len(faces))
            if mesh_generated:
                _write_obj(obj_path, vertices, faces, patch["patch_id"])
                combined_meshes.append(
                    (patch["patch_id"], vertices, faces)
                )
            confidence_values = confidence[valid_pixels]
            error_values = error[valid_pixels]
            triangle_areas = []
            for face in faces:
                triangle = points_image[face]
                triangle_areas.append(_triangle_area(triangle))
            trials.append(
                {
                    "patch_id": patch["patch_id"],
                    "parent_component_id": patch["parent_component_id"],
                    "query_count": patch["query_count"],
                    "source_points_xy_image": points_image.round(3).tolist(),
                    "faces": faces.tolist(),
                    "vertex_count": len(vertices),
                    "face_count": len(faces),
                    "mesh_generated": mesh_generated,
                    "topology": "open_surface_not_solid",
                    "nominal_relief_amplitude": NOMINAL_RELIEF_AMPLITUDE,
                    "projected_pixel_count": int(np.count_nonzero(valid_pixels)),
                    "dense_relief_mean_absolute_error": round(
                        float(np.mean(error_values)) if error_values.size else 0.0,
                        6,
                    ),
                    "dense_relief_rmse": round(
                        float(np.sqrt(np.mean(np.square(error_values))))
                        if error_values.size
                        else 0.0,
                        6,
                    ),
                    "mean_phase5_4_confidence": round(
                        float(np.mean(confidence_values))
                        if confidence_values.size
                        else 0.0,
                        6,
                    ),
                    "minimum_triangle_area_pixels2": round(
                        min(triangle_areas) if triangle_areas else 0.0, 6
                    ),
                    "finite_geometry": bool(
                        np.all(np.isfinite(vertices))
                        and np.all(np.isfinite(rendered[rendered_mask]))
                    ),
                    "source_evidence_state": patch["evidence_state"],
                    "semantic_label": "unknown_local_relief_surface_trial",
                    "semantic_claim_authorized": False,
                    "cad_fusion_authorized": False,
                    "artifact": (
                        {
                            "obj": _relative(obj_path, root),
                            "sha256": _sha256(obj_path),
                        }
                        if mesh_generated
                        else None
                    ),
                    "error_map_array": error,
                }
            )

        if not combined_meshes:
            raise RuntimeError(f"No local surfaces generated for {object_id}")
        combined_obj = object_dir / f"{object_id}_local_relief_trials.obj"
        _write_combined_obj(combined_obj, combined_meshes)
        combined_preview = object_dir / f"{object_id}_local_relief_trials_preview.png"
        _render_obj_preview(
            combined_obj,
            combined_preview,
            f"PHASE 5.12 - {object_id.upper()} LOCAL RELIEF TRIALS",
            minimum_xy_radius=0.08,
        )
        audit_path = object_dir / "local_relief_surface_audit.png"
        _render_object_audit(
            source,
            trials,
            combined_preview,
            audit_path,
            object_id,
        )
        audit_paths.append(audit_path)
        serializable_trials = []
        for trial in trials:
            serialized = dict(trial)
            serialized.pop("error_map_array")
            serializable_trials.append(serialized)
        object_report = {
            "schema_version": "lalitha_phase5_local_relief_surfaces_object_v1",
            "object_id": object_id,
            "split": "val",
            "eligible_patch_count": len(eligible),
            "generated_mesh_count": len(combined_meshes),
            "trials": serializable_trials,
            "claims": {
                "same_view_interpolation_accuracy_only": True,
                "physical_relief_accuracy_validated": False,
                "semantic_accuracy_validated": False,
                "cad_fusion_authorized": False,
            },
            "artifacts": {
                "combined_obj": _relative(combined_obj, root),
                "combined_obj_sha256": _sha256(combined_obj),
                "combined_preview": _relative(combined_preview, root),
                "audit": _relative(audit_path, root),
            },
        }
        object_report_path = object_dir / "report.json"
        object_report_path.write_text(
            json.dumps(object_report, indent=2) + "\n", encoding="utf-8"
        )
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "eligible_patch_count": len(eligible),
                "generated_mesh_count": len(combined_meshes),
                "failed_mesh_count": len(eligible) - len(combined_meshes),
                "vertex_count": sum(
                    item[1].shape[0] for item in combined_meshes
                ),
                "face_count": sum(item[2].shape[0] for item in combined_meshes),
                "mean_dense_relief_rmse": round(
                    float(
                        np.mean(
                            [
                                item["dense_relief_rmse"]
                                for item in serializable_trials
                                if item["mesh_generated"]
                            ]
                        )
                    ),
                    6,
                ),
                "all_generated_geometry_finite": all(
                    item["finite_geometry"]
                    for item in serializable_trials
                    if item["mesh_generated"]
                ),
                "phase5_11_eligible_count_reproduced": len(eligible)
                == subdivision_by_id[object_id][
                    "fit_review_eligible_patch_count"
                ],
                "artifacts": {
                    "report": _relative(object_report_path, root),
                    **object_report["artifacts"],
                },
            }
        )

    cohort_path = output / "validation_local_relief_surface_audit.png"
    _cohort_audit(audit_paths, cohort_path)
    eligible_count = sum(item["eligible_patch_count"] for item in objects)
    generated_count = sum(item["generated_mesh_count"] for item in objects)
    report = {
        "schema_version": "lalitha_phase5_local_relief_surfaces_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.12 reversible local relief-surface trials",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "source_view": "front",
        },
        "sources": {
            "subdivision": f"{SUBDIVISION_ROOT.as_posix()}/report.json",
            "phase5_4_relief_builder": "pipeline/build_phase5_stl1_relief_evidence.py",
            "phase1_features": PHASE1_REPORT.as_posix(),
        },
        "summary": {
            "object_count": len(objects),
            "eligible_patch_count": eligible_count,
            "generated_mesh_count": generated_count,
            "failed_mesh_count": eligible_count - generated_count,
            "vertex_count": sum(item["vertex_count"] for item in objects),
            "face_count": sum(item["face_count"] for item in objects),
            "mean_dense_relief_rmse": round(
                float(np.mean([item["mean_dense_relief_rmse"] for item in objects])),
                6,
            ),
        },
        "checks": {
            "fixed_validation_objects_only": all(
                item["split"] == "val" for item in objects
            ),
            "phase5_11_eligible_counts_reproduced": all(
                item["phase5_11_eligible_count_reproduced"] for item in objects
            ),
            "all_generated_geometry_finite": all(
                item["all_generated_geometry_finite"] for item in objects
            ),
            "all_generated_surfaces_are_open_obj_layers": True,
            "test_objects_not_processed": True,
            "exact_cad_unchanged": True,
            "no_step_files_generated": True,
        },
        "objects": objects,
        "decision": "retain_local_surfaces_as_reversible_review_trials_only",
        "claims": {
            "physical_relief_accuracy_validated": False,
            "semantic_accuracy_validated": False,
            "metric_scale_validated": False,
            "hidden_geometry_recovered": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Dense interpolation error is measured against the same retained image-depth evidence used to build the vertices, not physical relief ground truth.",
            "Sparse Delaunay triangles can bridge unseen detail inside a patch even when the outer patch gate passes.",
            "OBJ height is normalized for review and is not millimetric.",
            "Every generated surface is open, front-visible and separate from exact CAD.",
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
        default=Path("data/phase5_local_relief_surfaces_v1"),
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
