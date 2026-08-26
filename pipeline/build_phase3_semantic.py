"""Build a coherent Ring01 semantic proxy from Phase 2/3 evidence.

Dimensions are provisional assumptions until a physical reference is supplied.
The visual hull remains evidence; this jewellery-aware proxy supplies coherent
metal, stone, prongs, seat, shoulders, and cavity for downstream fitting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _cylinder_between(cq, start, end, radius):
    direction = end.sub(start)
    return cq.Workplane(obj=cq.Solid.makeCylinder(radius, direction.Length, start, direction.normalized()))


def build(phase2_2_dir: Path, phase3_dir: Path, output_dir: Path) -> dict:
    import cadquery as cq
    import trimesh

    phase2 = json.loads((phase2_2_dir / "phase2_2_components.json").read_text(encoding="utf-8"))
    phase3 = json.loads((phase3_dir / "phase3_reconstruction.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Provisional canonical scale.  Ratios are evidence-driven; millimetres
    # are an explicit assumption and must be replaced after calibration.
    hoop_major_radius = 10.0
    side_shank = phase2["views"]["side"]["components"]["shank"]["bbox_px"]
    shank_ratio = max(0.09, min(0.16, 0.5 * side_shank["width"] / max(1, side_shank["height"]) * 0.18))
    shank_radius = round(hoop_major_radius * shank_ratio, 3)
    front_stone = phase2["views"]["front"]["components"]["stone_amodal"]["bbox_px"]
    front_jewelry = phase2["views"]["front"]["components"]["jewelry"]["bbox_px"]
    stone_ratio = front_stone["width"] / max(1, front_jewelry["width"])
    stone_diameter = round(max(5.8, min(7.2, 20.0 * stone_ratio)), 3)
    stone_radius = stone_diameter / 2.0

    # CadQuery's default torus lies in XY. Rotate it into XZ so side/back show
    # the hoop and front/top look down the gemstone axis.
    metal = cq.Workplane(obj=cq.Solid.makeTorus(hoop_major_radius, shank_radius)).rotate((0, 0, 0), (1, 0, 0), 90)
    shoulders = None
    for sign in (-1.0, 1.0):
        start = cq.Vector(6.0 * sign, 0.0, 8.0)
        end = cq.Vector((stone_radius + 0.45) * sign, 0.0, 10.9)
        support = _cylinder_between(cq, start, end, max(0.85, shank_radius * 0.78))
        shoulders = support if shoulders is None else shoulders.union(support)
    metal = metal.union(shoulders)

    seat_z = 10.65
    seat_outer = cq.Workplane("XY").circle(stone_radius + 1.05).extrude(1.25).translate((0, 0, seat_z))
    seat_inner = cq.Workplane("XY").circle(max(0.9, stone_radius - 0.38)).extrude(2.0).translate((0, 0, seat_z - 0.25))
    metal = metal.union(seat_outer.cut(seat_inner))

    prong_radius = max(0.36, stone_radius * 0.115)
    prong_height = 4.4
    prongs = None
    prong_centers = []
    for angle in (45, 135, 225, 315):
        radians = math.radians(angle)
        x = (stone_radius + 0.30) * math.cos(radians)
        y = (stone_radius + 0.30) * math.sin(radians)
        prong = cq.Workplane("XY").circle(prong_radius).extrude(prong_height).translate((x, y, seat_z + 0.15))
        prongs = prong if prongs is None else prongs.union(prong)
        prong_centers.append([round(x, 4), round(y, 4), round(seat_z + 0.15, 4)])
    metal = metal.union(prongs)

    cavity_radius = max(0.8, stone_radius - 0.22)
    cavity = cq.Workplane("XY").circle(cavity_radius).extrude(3.5).translate((0, 0, seat_z + 0.05))
    metal = metal.cut(cavity)

    pavilion = (
        cq.Workplane("XY")
        .workplane(offset=11.15)
        .circle(stone_radius * 0.16)
        .workplane(offset=2.05)
        .circle(stone_radius)
        .loft()
    )
    crown = (
        cq.Workplane("XY")
        .workplane(offset=13.2)
        .circle(stone_radius)
        .workplane(offset=1.30)
        .circle(stone_radius * 0.55)
        .loft()
    )
    gemstone = pavilion.union(crown)
    assembly = cq.Compound.makeCompound([metal.val(), gemstone.val()])

    paths = {
        "metal_step": output_dir / "ring01_phase3_metal.step",
        "stone_step": output_dir / "ring01_phase3_stone.step",
        "assembly_step": output_dir / "ring01_phase3_semantic.step",
        "assembly_stl": output_dir / "ring01_phase3_semantic.stl",
    }
    cq.exporters.export(metal, str(paths["metal_step"]))
    cq.exporters.export(gemstone, str(paths["stone_step"]))
    cq.exporters.export(assembly, str(paths["assembly_step"]))
    cq.exporters.export(assembly, str(paths["assembly_stl"]))

    mesh = trimesh.load(paths["assembly_stl"], force="mesh")
    obj_path = output_dir / "ring01_phase3_semantic.obj"
    glb_path = output_dir / "ring01_phase3_semantic.glb"
    mesh.export(obj_path)
    mesh.export(glb_path)
    try:
        from pipeline.reconstruct_phase3 import _render_preview
    except ModuleNotFoundError:
        from reconstruct_phase3 import _render_preview

    preview_path = _render_preview(mesh, output_dir)
    result = {
        "stage": "phase3_jewellery_aware_semantic_proxy",
        "source_visual_hull": str(phase3_dir / "phase3_reconstruction.json"),
        "scale_status": "provisional_uncalibrated_millimetre_assumption",
        "assumptions": {
            "hoop_major_radius_mm": hoop_major_radius,
            "shank_radius_mm": shank_radius,
            "stone_diameter_mm": stone_diameter,
            "warning": "These values are fitting seeds, not measured dimensions.",
        },
        "topology": {
            "one_shank": True,
            "one_stone": True,
            "prong_count": 4,
            "stone_seat": True,
            "boolean_cavity": True,
            "prong_centers": prong_centers,
        },
        "validation": {
            "metal_solid_count": len(metal.solids().vals()),
            "stone_solid_count": len(gemstone.solids().vals()),
            "assembly_solid_count": len(assembly.Solids()),
            "metal_is_single_solid": len(metal.solids().vals()) == 1,
            "stone_is_single_solid": len(gemstone.solids().vals()) == 1,
        },
        "artifacts": {key: str(value) for key, value in paths.items()} | {
            "assembly_obj": str(obj_path),
            "assembly_glb": str(glb_path),
            "preview": preview_path,
        },
    }
    (output_dir / "phase3_semantic_proxy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-2-dir", type=Path, default=Path("data/ring01_phase2_2"))
    parser.add_argument("--phase3-dir", type=Path, default=Path("data/ring01_phase3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_phase3/semantic"))
    args = parser.parse_args()
    print(json.dumps(build(args.phase2_2_dir, args.phase3_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
