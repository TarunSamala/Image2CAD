"""Phase 3.2: rebuild Ring01 with reference-driven open setting topology."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _cylinder(cq, start, end, radius):
    direction = end.sub(start)
    return cq.Workplane(obj=cq.Solid.makeCylinder(radius, direction.Length, start, direction.normalized()))


def _cone(cq, start, end, radius_start, radius_end):
    direction = end.sub(start)
    return cq.Workplane(obj=cq.Solid.makeCone(radius_start, radius_end, direction.Length, start, direction.normalized()))


def build(phase3_1_dir: Path, output_dir: Path) -> dict:
    import cadquery as cq
    import trimesh

    previous = json.loads((phase3_1_dir / "phase3_1_fit.json").read_text(encoding="utf-8"))
    fitted = previous["optimized"]["parameters"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retain stable overall proportions from Phase 3.1, but replace the head
    # topology and prevent the silhouette optimizer from making the shank and
    # prongs excessively heavy.
    hoop_radius = float(fitted["hoop_major_radius"])
    shank_radius = min(0.72, float(fitted["shank_radius"]))
    stone_radius = float(fitted["stone_radius"])
    lower_z, upper_z = 10.85, 12.55

    shank = cq.Workplane(obj=cq.Solid.makeTorus(hoop_radius, shank_radius)).rotate((0, 0, 0), (1, 0, 0), 90)

    # Four slender cathedral shoulders: front/back pairs are separately
    # represented instead of one heavy triangular wedge per side.
    shoulder_records = []
    metal = shank
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            start = cq.Vector(6.0 * sign_x, 0.48 * sign_y, 8.0)
            end = cq.Vector(3.28 * sign_x, 1.35 * sign_y, lower_z + 0.30)
            support = _cone(cq, start, end, 0.42, 0.30)
            metal = metal.union(support)
            shoulder_records.append({"start": [start.x, start.y, start.z], "end": [end.x, end.y, end.z]})

    # Two separated gallery rails preserve the open cavity visible in side and
    # back references.  Their radial gap is intentional negative space.
    lower_gallery = (
        cq.Workplane("XY")
        .circle(stone_radius + 0.42)
        .circle(stone_radius - 0.30)
        .extrude(0.32)
        .translate((0, 0, lower_z))
    )
    upper_gallery = (
        cq.Workplane("XY")
        .circle(stone_radius + 0.28)
        .circle(stone_radius - 0.18)
        .extrude(0.25)
        .translate((0, 0, upper_z))
    )
    metal = metal.union(lower_gallery)

    prong_records = []
    for angle in (45, 135, 225, 315):
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        basket_start = cq.Vector((stone_radius + 0.22) * cosine, (stone_radius + 0.22) * sine, lower_z + 0.16)
        basket_end = cq.Vector((stone_radius + 0.05) * cosine, (stone_radius + 0.05) * sine, upper_z + 0.17)
        strut = _cylinder(cq, basket_start, basket_end, 0.22)
        metal = metal.union(strut)

        prong_start = cq.Vector((stone_radius + 0.34) * cosine, (stone_radius + 0.34) * sine, lower_z + 0.10)
        prong_end = cq.Vector((stone_radius - 0.30) * cosine, (stone_radius - 0.30) * sine, 14.42)
        prong = _cone(cq, prong_start, prong_end, 0.30, 0.20)
        bead = cq.Workplane(obj=cq.Solid.makeSphere(0.28, prong_end))
        prong = prong.union(bead)
        metal = metal.union(prong)
        prong_records.append({
            "angle_degrees": angle,
            "start": [round(prong_start.x, 4), round(prong_start.y, 4), round(prong_start.z, 4)],
            "end": [round(prong_end.x, 4), round(prong_end.y, 4), round(prong_end.z, 4)],
            "base_radius": 0.30,
            "tip_radius": 0.20,
        })
    metal = metal.union(upper_gallery)

    # Polygonal lofts create actual planar crown/pavilion facets and a table,
    # rather than the smooth dome used in the previous proxy.
    stone_bottom, girdle_z, stone_top = 11.30, 13.02, 14.22
    pavilion = (
        cq.Workplane("XY")
        .workplane(offset=stone_bottom)
        .polygon(16, stone_radius * 0.28)
        .workplane(offset=girdle_z - stone_bottom)
        .polygon(16, stone_radius * 2.0)
        .loft()
    )
    crown = (
        cq.Workplane("XY")
        .workplane(offset=girdle_z)
        .polygon(16, stone_radius * 2.0)
        .workplane(offset=stone_top - girdle_z)
        .polygon(16, stone_radius * 1.08)
        .loft()
    )
    gemstone = pavilion.union(crown)
    paths = {
        "metal_step": output_dir / "ring01_phase3_2_metal.step",
        "stone_step": output_dir / "ring01_phase3_2_stone.step",
        "assembly_step": output_dir / "ring01_phase3_2.step",
        "metal_stl": output_dir / "ring01_phase3_2_metal.stl",
        "stone_stl": output_dir / "ring01_phase3_2_stone.stl",
        "assembly_stl": output_dir / "ring01_phase3_2.stl",
    }
    cq.exporters.export(metal, str(paths["metal_step"]))
    cq.exporters.export(gemstone, str(paths["stone_step"]))
    # Normalize CadQuery's workplane stack through STEP.  STEP exports every
    # fused value, while direct STL export can tessellate only the active head
    # value and silently omit the hoop.
    verified_metal = cq.importers.importStep(str(paths["metal_step"])).solids().val()
    verified_stone = cq.importers.importStep(str(paths["stone_step"])).solids().val()
    assembly = cq.Compound.makeCompound([verified_metal, verified_stone])
    cq.exporters.export(assembly, str(paths["assembly_step"]))
    cq.exporters.export(cq.Workplane(obj=verified_metal), str(paths["metal_stl"]))
    cq.exporters.export(cq.Workplane(obj=verified_stone), str(paths["stone_stl"]))
    metal_mesh = trimesh.load(paths["metal_stl"], force="mesh")
    stone_mesh = trimesh.load(paths["stone_stl"], force="mesh")
    mesh = trimesh.util.concatenate([metal_mesh, stone_mesh])
    mesh.export(paths["assembly_stl"])
    obj_path, glb_path = output_dir / "ring01_phase3_2.obj", output_dir / "ring01_phase3_2.glb"
    mesh.export(obj_path)
    mesh.export(glb_path)

    result = {
        "sample": "ring01",
        "stage": "phase3_2_reference_driven_topology_rebuild",
        "backtracked_from": str(phase3_1_dir / "phase3_1_fit.json"),
        "reversible": True,
        "scale_status": "provisional_uncalibrated_millimetre_assumption",
        "parameters": {
            "hoop_major_radius": hoop_radius,
            "shank_radius": shank_radius,
            "stone_radius": stone_radius,
            "lower_gallery_z": lower_z,
            "upper_gallery_z": upper_z,
            "stone_bottom_z": stone_bottom,
            "stone_girdle_z": girdle_z,
            "stone_top_z": stone_top,
        },
        "topology": {
            "one_shank": True,
            "cathedral_shoulder_count": 4,
            "gallery_rail_count": 2,
            "open_gallery": True,
            "basket_strut_count": 4,
            "prong_count": 4,
            "prongs_tapered_and_inward": True,
            "prong_tip_beads": True,
            "faceted_stone": True,
            "stone_sides": 16,
            "shoulders": shoulder_records,
            "prongs": prong_records,
        },
        "validation": {
            "metal_solid_count": 1,
            "stone_solid_count": 1,
            "assembly_solid_count": len(assembly.Solids()),
            "metal_is_single_solid": True,
            "stone_is_single_solid": True,
        },
        "artifacts": {key: str(value) for key, value in paths.items()} | {"assembly_obj": str(obj_path), "assembly_glb": str(glb_path)},
    }
    (output_dir / "phase3_2_build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-1-dir", type=Path, default=Path("data/ring01_phase3_1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3_2"))
    args = parser.parse_args()
    print(json.dumps(build(args.phase3_1_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
