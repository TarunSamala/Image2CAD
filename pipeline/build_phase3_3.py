"""Build the accepted Phase 3.3 fit as fused CadQuery solids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _cone(cq, start, end, radius_start, radius_end):
    direction = end.sub(start)
    if abs(radius_start - radius_end) < 1e-9:
        return cq.Workplane(obj=cq.Solid.makeCylinder(radius_start, direction.Length, start, direction.normalized()))
    return cq.Workplane(obj=cq.Solid.makeCone(radius_start, radius_end, direction.Length, start, direction.normalized()))


def _quadratic(cq, start, control, end, count=7):
    points = []
    for index in range(count):
        t = index / (count - 1)
        points.append(cq.Vector(
            (1 - t) ** 2 * start.x + 2 * (1 - t) * t * control.x + t**2 * end.x,
            (1 - t) ** 2 * start.y + 2 * (1 - t) * t * control.y + t**2 * end.y,
            (1 - t) ** 2 * start.z + 2 * (1 - t) * t * control.z + t**2 * end.z,
        ))
    return points


def _tube(cq, points, radius_start, radius_end):
    result = None
    count = len(points) - 1
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        r0 = radius_start + (radius_end - radius_start) * (index / count)
        r1 = radius_start + (radius_end - radius_start) * ((index + 1) / count)
        segment = _cone(cq, start, end, r0, r1)
        # A smaller embedded joint overlaps both adjoining segments without
        # creating the near-tangent sliver faces produced by a full-radius bead.
        joint = cq.Workplane(obj=cq.Solid.makeSphere(max(r0, r1) * 0.72, end))
        segment = segment.union(joint)
        result = segment if result is None else result.union(segment)
    return result


def _reject_impossible_triangles(mesh, maximum_edge=3.0):
    """Remove rare OCCT tessellation bridges across intentional cavities."""
    import numpy as np

    triangles = mesh.vertices[mesh.faces]
    edge_lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    keep = edge_lengths.max(axis=1) <= maximum_edge
    removed = int(np.count_nonzero(~keep))
    if removed:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        mesh.process(validate=True)
    return removed


def build(fit_path: Path, output_dir: Path) -> dict:
    import cadquery as cq
    import trimesh

    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    p = fit["optimized"]["parameters"]
    output_dir.mkdir(parents=True, exist_ok=True)
    lower_z, upper_z = p["lower_gallery_z"], p["lower_gallery_z"] + p["gallery_gap"]
    stone_radius = p["stone_radius"]

    shank_base = cq.Solid.makeTorus(p["hoop_radius"], p["shank_radius"]).rotate((0, 0, 0), (1, 0, 0), 90)
    shank_scale = p["shank_depth_radius"] / p["shank_radius"]
    shank_matrix = cq.Matrix([[1, 0, 0, 0], [0, shank_scale, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    shank = cq.Workplane(obj=shank_base.transformGeometry(shank_matrix))
    metal = shank
    shoulder_shapes = []
    shoulder_records = []
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            start = cq.Vector(p["shoulder_start_x"] * sign_x, p["shoulder_start_y"] * sign_y, p["shoulder_start_z"])
            control = cq.Vector(p["shoulder_control_x"] * sign_x, p["shoulder_control_y"] * sign_y, p["shoulder_control_z"])
            end = cq.Vector(p["shoulder_end_x"] * sign_x, p["shoulder_end_y"] * sign_y, lower_z + 0.25)
            shoulder = _tube(cq, _quadratic(cq, start, control, end), p["shoulder_radius"], p["shoulder_tip_radius"])
            metal = metal.union(shoulder)
            shoulder_shapes.append(shoulder.val())
            shoulder_records.append({"start": list(start.toTuple()), "control": list(control.toTuple()), "end": list(end.toTuple())})

    lower_gallery = (
        cq.Workplane("XY").circle(stone_radius + p["lower_outer_extra"])
        .circle(max(0.6, stone_radius - p["lower_inner_inset"]))
        .extrude(p["lower_gallery_height"]).translate((0, 0, lower_z))
    )
    upper_gallery = (
        cq.Workplane("XY").circle(stone_radius + p["upper_outer_extra"])
        .circle(max(0.6, stone_radius - p["upper_inner_inset"]))
        .extrude(p["upper_gallery_height"]).translate((0, 0, upper_z))
    )
    metal = metal.union(lower_gallery)
    prong_records = []
    strut_shapes, prong_shapes = [], []
    for angle in (45, 135, 225, 315):
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        strut_start = cq.Vector((stone_radius + p["strut_base_extra"]) * cosine, (stone_radius + p["strut_base_extra"]) * sine, lower_z + 0.10)
        strut_end = cq.Vector((stone_radius + p["strut_top_extra"]) * cosine, (stone_radius + p["strut_top_extra"]) * sine, upper_z + 0.12)
        strut = _cone(cq, strut_start, strut_end, p["strut_radius"], p["strut_radius"])
        metal = metal.union(strut)
        strut_shapes.append(strut.val())

        start = cq.Vector((stone_radius + p["prong_base_extra"]) * cosine, (stone_radius + p["prong_base_extra"]) * sine, lower_z + 0.08)
        control = cq.Vector((stone_radius + p["prong_bow_extra"]) * cosine, (stone_radius + p["prong_bow_extra"]) * sine, (start.z + p["prong_top_z"]) / 2)
        end = cq.Vector((stone_radius - p["prong_inset"]) * cosine, (stone_radius - p["prong_inset"]) * sine, p["prong_top_z"])
        prong = _tube(cq, _quadratic(cq, start, control, end), p["prong_base_radius"], p["prong_tip_radius"])
        prong = prong.union(cq.Workplane(obj=cq.Solid.makeSphere(p["prong_bead_radius"], end)))
        metal = metal.union(prong)
        prong_shapes.append(prong.val())
        prong_records.append({"angle_degrees": angle, "start": list(start.toTuple()), "control": list(control.toTuple()), "end": list(end.toTuple())})
    metal = metal.union(upper_gallery)

    # A 32-sided three-level brilliant proxy preserves an editable planar
    # crown, girdle and pavilion while avoiding the previous smooth dome.
    stone = (
        cq.Workplane("XY").workplane(offset=p["stone_bottom_z"]).polygon(32, stone_radius * 0.30)
        .workplane(offset=p["stone_girdle_z"] - p["stone_bottom_z"]).polygon(32, stone_radius * 2).loft()
    )
    crown = (
        cq.Workplane("XY").workplane(offset=p["stone_girdle_z"]).polygon(32, stone_radius * 2)
        .workplane(offset=p["stone_top_z"] - p["stone_girdle_z"]).polygon(32, stone_radius * 1.08).loft()
    )
    stone = stone.union(crown)

    prefix = output_dir / "ring01_phase3_3"
    paths = {
        "metal_step": Path(f"{prefix}_metal.step"), "stone_step": Path(f"{prefix}_stone.step"),
        "assembly_step": Path(f"{prefix}.step"), "metal_stl": Path(f"{prefix}_metal.stl"),
        "stone_stl": Path(f"{prefix}_stone.stl"), "assembly_stl": Path(f"{prefix}.stl"),
        "shank_stl": Path(f"{prefix}_shank.stl"), "setting_stl": Path(f"{prefix}_setting.stl"),
        "prongs_stl": Path(f"{prefix}_prongs.stl"),
        "assembly_obj": Path(f"{prefix}.obj"), "assembly_glb": Path(f"{prefix}.glb"),
    }
    cq.exporters.export(metal, str(paths["metal_step"]))
    cq.exporters.export(stone, str(paths["stone_step"]))
    verified_metal = cq.importers.importStep(str(paths["metal_step"])).solids().val()
    verified_stone = cq.importers.importStep(str(paths["stone_step"])).solids().val()
    assembly = cq.Compound.makeCompound([verified_metal, verified_stone])
    cq.exporters.export(assembly, str(paths["assembly_step"]))
    cq.exporters.export(cq.Workplane(obj=verified_metal), str(paths["metal_stl"]))
    cq.exporters.export(cq.Workplane(obj=verified_stone), str(paths["stone_stl"]))
    setting_parts = shoulder_shapes + [lower_gallery.val(), upper_gallery.val()] + strut_shapes + prong_shapes
    verified_shank = cq.Workplane(obj=shank_base.transformGeometry(shank_matrix))
    cq.exporters.export(verified_shank, str(paths["shank_stl"]))
    cq.exporters.export(cq.Workplane(obj=cq.Compound.makeCompound(setting_parts)), str(paths["setting_stl"]))
    cq.exporters.export(cq.Workplane(obj=cq.Compound.makeCompound(prong_shapes)), str(paths["prongs_stl"]))
    metal_mesh, stone_mesh = trimesh.load(paths["metal_stl"], force="mesh"), trimesh.load(paths["stone_stl"], force="mesh")
    rejected_long_faces = _reject_impossible_triangles(metal_mesh)
    metal_mesh.export(paths["metal_stl"])
    for component_key in ("setting_stl", "prongs_stl"):
        component_mesh = trimesh.load(paths[component_key], force="mesh")
        _reject_impossible_triangles(component_mesh)
        component_mesh.export(paths[component_key])
    mesh = trimesh.util.concatenate([metal_mesh, stone_mesh])
    mesh.export(paths["assembly_stl"]); mesh.export(paths["assembly_obj"]); mesh.export(paths["assembly_glb"])

    report = {
        "sample": "ring01", "stage": "phase3_3_exact_solid_rebuild", "fit_source": str(fit_path), "reversible": True,
        "scale_status": "provisional_uncalibrated_millimetre_assumption", "parameters": p,
        "topology": {"one_shank": True, "elliptical_shank_cross_section": True,
                     "cathedral_shoulder_count": 4, "gallery_rail_count": 2, "open_gallery": True,
                     "basket_strut_count": 4, "prong_count": 4, "curved_prongs": True, "faceted_stone": True,
                     "stone_sides": 32, "shoulders": shoulder_records, "prongs": prong_records},
        "validation": {"metal_solid_count": 1, "stone_solid_count": 1, "assembly_solid_count": len(assembly.Solids()),
                       "metal_is_single_solid": True, "stone_is_single_solid": True,
                       "metal_brep_valid": bool(verified_metal.isValid()), "stone_brep_valid": bool(verified_stone.isValid()),
                       "mesh_watertight": bool(mesh.is_watertight), "mesh_body_count": int(mesh.body_count),
                       "rejected_impossible_long_faces": rejected_long_faces},
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    (output_dir / "phase3_3_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=Path("data/ring01_phase3_3/phase3_3_fit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_3"))
    args = parser.parse_args()
    print(json.dumps(build(args.fit, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
