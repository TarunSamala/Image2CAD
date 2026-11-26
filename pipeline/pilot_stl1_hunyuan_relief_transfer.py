"""Try bounded front relief on the four-view hulls of the STL-1 Hunyuan pilot."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from trimesh.sample import sample_surface

from pipeline.benchmark_stl1_four_view_holdout import (
    AUDIT_SIZE, MARGIN, OUTPUT as HULL_ROOT, _load_upright, _normalize_mask, _score,
)
from pipeline.pilot_stl1_hunyuan import OBJECTS, OUTPUT as GENERATOR_ROOT
from pipeline.reconstruct_dataset_phase3 import _cell, _error_overlay, _manifest, _read
from pipeline.refine_phase3_1 import _jewelry_transform
from pipeline.run_saraswati_blind_benchmark import (
    _align, _foreground_mask, _load_mesh, _normalize, _sha256, _surface_metrics, _write_json,
)
from pipeline.train_dataset_phases import _binary_iou
from pipeline.validate_phase3_2 import _basis_camera, _clean_render


OUTPUT = Path("dataset/phase_runs/v1/stl1_hunyuan_relief_transfer_v1")
FRAME = (MARGIN, MARGIN, AUDIT_SIZE - MARGIN - 1, AUDIT_SIZE - MARGIN - 1)
FRONT_AXIS_FRACTION = 0.30
MAX_DISPLACEMENT_FRACTION = 0.018
MAX_SOURCE_SILHOUETTE_CHANGE = 0.01
INTERIOR_MARGIN_PX = 4.0


def _front_depth(mesh, pixels, seed):
    """Sample the front envelope and keep the distance to the nearest sample."""
    points, _ = sample_surface(mesh, 650_000, seed=seed)
    x, z = np.rint(pixels(points)).astype(int).T
    valid = (x >= 0) & (x < AUDIT_SIZE) & (z >= 0) & (z < AUDIT_SIZE)
    depth = np.full((AUDIT_SIZE, AUDIT_SIZE), np.inf)
    np.minimum.at(depth, (z[valid], x[valid]), points[valid, 1])
    distance, nearest = distance_transform_edt(~np.isfinite(depth), return_indices=True)
    return depth[tuple(nearest)], distance <= 1.5


def _masked_blur(values, mask, sigma):
    weights = cv2.GaussianBlur(mask.astype(float), (0, 0), sigma)
    blurred = cv2.GaussianBlur(np.where(mask, values, 0), (0, 0), sigma)
    return blurred / np.maximum(weights, 1e-8)


def _displacements(base, donor):
    vertices = np.asarray(base.vertices)
    extent = float(base.extents.max())
    center = (base.bounds[0] + base.bounds[1]) / 2
    scale = (AUDIT_SIZE - 2 * MARGIN - 1) / float(base.extents[[0, 2]].max())

    def pixels(points):
        return (points[:, [0, 2]] - center[[0, 2]]) * [scale, -scale] + (AUDIT_SIZE - 1) / 2

    baseline_depth, base_valid = _front_depth(base, pixels, seed=4053)
    donor_depth, donor_valid = _front_depth(donor, pixels, seed=4054)
    overlap = base_valid & donor_valid
    x_distance = distance_transform_edt(overlap)
    aligned = np.abs(donor_depth - baseline_depth) <= 0.12 * extent
    region = overlap & aligned & (baseline_depth <= base.bounds[0, 1] + FRONT_AXIS_FRACTION * base.extents[1])
    # Remove broad depth and pose changes. Transfer only local variation from
    # the donor, with a taper at both the support edge and front silhouette.
    coarse = _masked_blur(donor_depth, region, sigma=7.0)
    local = _masked_blur(donor_depth, region, sigma=1.3) - coarse
    boundary_distance = np.minimum(x_distance, distance_transform_edt(region))
    taper = np.clip((boundary_distance - INTERIOR_MARGIN_PX) / 5, 0, 1)
    limit = extent * MAX_DISPLACEMENT_FRACTION
    field = np.clip(local, -limit, limit) * taper
    px, py = pixels(vertices).T
    grid = [py, px]
    front = map_coordinates(baseline_depth, grid, order=1, mode="nearest")
    exposed = (np.abs(vertices[:, 1] - front) <= 0.014 * extent) & (base.vertex_normals[:, 1] < -0.45)
    exposed &= vertices[:, 1] <= base.bounds[0, 1] + FRONT_AXIS_FRACTION * base.extents[1]
    displacement = np.where(exposed, map_coordinates(field, grid, order=1, mode="nearest"), 0.0)
    return displacement, field


def _project_fixed(mesh, camera, transform):
    scale, (u_center, v_center), (x0, y0, x1, y1) = transform
    vertices = np.asarray(mesh.vertices)
    x = (vertices @ camera.right - u_center) * scale + (x0 + x1) / 2
    y = -(vertices @ camera.up - v_center) * scale + (y0 + y1) / 2
    result = np.zeros((AUDIT_SIZE, AUDIT_SIZE), np.uint8)
    for triangle in np.rint(np.column_stack([x, y])[mesh.faces]).astype(np.int32):
        cv2.fillConvexPoly(result, triangle, 1)
    return result > 0


def _make_candidate(base, displacement):
    cameras = {
        view: _basis_camera(view, direction) for view, direction in (
            ("front", (0, -1, 0)), ("top", (0, 0, 1)),
            ("left", (-1, 0, 0)), ("right", (1, 0, 0)),
        )
    }
    transforms = {view: _jewelry_transform(base, camera, FRAME) for view, camera in cameras.items()}
    projections = {view: _project_fixed(base, camera, transforms[view]) for view, camera in cameras.items()}
    trials = []
    for strength in (1.0, 0.5, 0.25):
        candidate = base.copy()
        candidate.vertices[:, 1] += strength * displacement
        changed = np.abs(strength * displacement) > 1e-7
        face_normal_dot = np.einsum("ij,ij->i", base.face_normals, candidate.face_normals)
        silhouettes = {}
        for view, camera in cameras.items():
            projection = _project_fixed(candidate, camera, transforms[view])
            silhouettes[view] = {
                "overlap_with_baseline": round(_binary_iou(projections[view], projection), 6),
                "baseline_openings": _score(projections[view], projection)["enclosed_openings_largest_first"],
            }
        guard = bool(
            changed.any() and np.all(face_normal_dot > 0) and candidate.is_watertight
            and candidate.is_winding_consistent and candidate.body_count == base.body_count == 1
            and candidate.is_volume and np.array_equal(base.faces, candidate.faces)
            and all(item["overlap_with_baseline"] >= 1 - MAX_SOURCE_SILHOUETTE_CHANGE
                    and all(hole["projected_empty_fraction"] >= .99 for hole in item["baseline_openings"])
                    for item in silhouettes.values())
        )
        trials.append({
            "strength": strength, "guard_passed": guard, "changed_vertices": int(changed.sum()),
            "max_displacement_display_units": float(np.abs(strength * displacement).max()),
            "minimum_face_normal_dot": float(face_normal_dot.min()), "input_axis_projection_checks": silhouettes,
        })
        if guard:
            return candidate, trials
    return base, trials


def _compare_head_surfaces(base, candidate, proxy):
    """Use a shared proxy alignment to isolate geometry changes at the front."""
    base_points = sample_surface(base, 80_000, seed=731)[0]
    base_norm, center, scale = _normalize(base_points)
    proxy_norm = _normalize(sample_surface(proxy, 80_000, seed=732)[0])[0]
    aligned, fit = _align(proxy_norm, base_norm)
    cutoff = (base.bounds[0, 1] + FRONT_AXIS_FRACTION * base.extents[1] - center[1]) / scale
    head_reference = aligned[aligned[:, 1] <= cutoff]
    if len(head_reference) < 100:
        raise ValueError("Paired OBJ has insufficient front samples under the shared alignment")
    candidate_norm = (sample_surface(candidate, 80_000, seed=731)[0] - center) / scale
    return {
        "method": "one proxy-to-baseline alignment; same normalized front-depth slab for both meshes",
        "normalized_front_depth_cutoff": float(cutoff),
        "proxy_points_in_slab": len(head_reference),
        "alignment": fit,
        "baseline": _surface_metrics(base_norm[base_norm[:, 1] <= cutoff], head_reference),
        "relief_candidate": _surface_metrics(candidate_norm[candidate_norm[:, 1] <= cutoff], head_reference),
    }


def run():
    hull = json.loads((HULL_ROOT / "report.json").read_text(encoding="utf-8"))
    records = {row["object_id"]: row for row in hull["objects"]}
    prepared = {row["object_id"]: row for row in _manifest(Path("dataset/prepared_v1"))}
    azimuth, elevation = np.deg2rad([hull["camera"]["azimuth"], hull["camera"]["elevation"]])
    camera = _basis_camera("stl1_iso", (
        np.sin(azimuth) * np.cos(elevation), -np.cos(azimuth) * np.cos(elevation), np.sin(elevation),
    ))
    reports = []
    for object_id in OBJECTS:
        row = records[object_id]
        output = OUTPUT / object_id
        output.mkdir(parents=True, exist_ok=True)
        source = GENERATOR_ROOT / object_id
        comparison = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
        base_path = Path(row["mesh"])
        hull_generation = json.loads(Path(row["generation_record"]).read_text(encoding="utf-8"))
        donor_path = source / "hunyuan_shape.obj"
        donor_hash = _sha256(donor_path)
        if donor_hash != comparison["candidate_sha256"] or _sha256(base_path) != hull_generation["proposal"]["sha256"]:
            raise ValueError(f"Frozen mesh changed for {object_id}")
        base = _load_upright(base_path)
        donor = _load_mesh(donor_path)
        fit = comparison["hunyuan_alignment"]
        _, center, extent = _normalize(sample_surface(base, 12000, seed=456)[0])
        normalized = (np.asarray(donor.vertices) - fit["source_center"]) / fit["source_extent"]
        donor.vertices = (normalized @ np.asarray(fit["rotation_source_to_target"]).T
                          + fit["normalized_translation"]) * extent + center
        displacement, field = _displacements(base, donor)
        candidate, trials = _make_candidate(base, displacement)
        mesh_path = output / "relief_candidate.obj"
        original_frame = candidate.copy()
        original_frame.vertices *= [1, -1, -1]
        original_frame.export(mesh_path)
        exported = _load_upright(mesh_path)
        if (not exported.is_watertight or not exported.is_winding_consistent
                or exported.body_count != base.body_count or len(exported.faces) != len(base.faces)):
            raise ValueError(f"Export changed connectivity for {object_id}")

        # Read the angled image only after the candidate is fixed and exported.
        iso_path = Path(row["iso_path"])
        if _sha256(iso_path) != row["iso_sha256"]:
            raise ValueError(f"Frozen angled image changed for {object_id}")
        target = _normalize_mask(_foreground_mask(iso_path)[0])
        transform = _jewelry_transform(base, camera, FRAME)
        front = _basis_camera("front", (0, -1, 0))
        front_transform = _jewelry_transform(base, front, FRAME)
        panels = []
        metrics = {}
        for name, mesh in (("baseline", base), ("relief_candidate", exported)):
            projection = _project_fixed(mesh, camera, transform)
            iso_render = cv2.cvtColor(_clean_render(mesh, camera, target.shape, transform), cv2.COLOR_BGR2GRAY)
            front_render = cv2.cvtColor(_clean_render(mesh, front, target.shape, front_transform), cv2.COLOR_BGR2GRAY)
            metrics[name] = _score(target, projection)
            panels.append(np.hstack([
                _cell(_read(iso_path), "UNUSED ANGLED IMAGE"),
                _cell(front_render, f"{name.upper()} / FRONT"),
                _cell(iso_render, f"{name.upper()} / ANGLED"),
                _cell(_error_overlay(target, projection), "ANGLED SILHOUETTE"),
            ]))
        limit = float(base.extents.max()) * MAX_DISPLACEMENT_FRACTION
        field_image = cv2.applyColorMap(np.uint8(np.clip(0.5 + 0.5 * field / limit, 0, 1) * 255), cv2.COLORMAP_TURBO)
        field_image[np.abs(field) < 1e-7] = 245
        if not cv2.imwrite(str(output / "comparison.png"), np.vstack(panels)):
            raise OSError(f"Could not save comparison for {object_id}")
        if not cv2.imwrite(str(output / "donor_relief_field.png"), field_image):
            raise OSError(f"Could not save relief field for {object_id}")
        source_dir = Path("dataset/STL-1") / prepared[object_id]["source_object_name"]
        proxies = list(source_dir.glob("*.obj"))
        if len(proxies) != 1:
            raise ValueError(f"Expected exactly one paired OBJ for {object_id}")
        proxy_comparison = _compare_head_surfaces(base, exported, _load_mesh(proxies[0]))
        proxy_comparison["proxy_sha256"] = _sha256(proxies[0])
        result = {
            "object_id": object_id, "split": row["split"], "mesh": str(mesh_path),
            "sha256": _sha256(mesh_path), "baseline_sha256": hull_generation["proposal"]["sha256"],
            "donor_sha256": donor_hash, "angled_ref_sha256": row["iso_sha256"],
            "changed_from_baseline": candidate is not base, "trials": trials,
            "watertight": bool(exported.is_watertight), "connected_bodies": int(exported.body_count),
            "frozen_angle_evaluation": metrics,
            "paired_obj_head_evaluation": proxy_comparison,
        }
        _write_json(output / "report.json", result)
        reports.append(result)
        print(object_id, trials[-1]["changed_vertices"], metrics["relief_candidate"]["silhouette_iou"], flush=True)
    _write_json(OUTPUT / "report.json", {
        "experiment": "bounded front-depth high frequencies on fixed hull topology",
        "settings": {"front_depth_fraction": FRONT_AXIS_FRACTION, "max_displacement_fraction": MAX_DISPLACEMENT_FRACTION,
                     "min_pixel_distance_to_region_edge": INTERIOR_MARGIN_PX,
                     "max_input_axis_silhouette_change": MAX_SOURCE_SILHOUETTE_CHANGE},
        "limits": {"angled_image_used_for_transfer": False, "proxy_obj_used_for_transfer": False,
                   "self_intersection_tested": False, "physical_scale_validated": False},
        "objects": reports,
    })


if __name__ == "__main__":
    run()
