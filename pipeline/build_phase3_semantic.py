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


def _cone_between(cq, start, end, radius_start, radius_end):
    direction = end.sub(start)
    return cq.Workplane(obj=cq.Solid.makeCone(radius_start, radius_end, direction.Length, start, direction.normalized()))


def build(phase2_2_dir: Path, phase3_dir: Path, output_dir: Path, fit_report: Path | None = None) -> dict:
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
    parameters = {
        "hoop_major_radius": hoop_major_radius,
        "shank_radius": shank_radius,
        "stone_radius": stone_radius,
        "shoulder_end_x": stone_radius + 0.45,
        "shoulder_radius": max(0.85, shank_radius * 0.78),
        "seat_z": 10.65,
        "seat_inner_inset": 0.38,
        "seat_outer_extra": 1.05,
        "seat_height": 1.25,
        "prong_radius": max(0.36, stone_radius * 0.115),
        "prong_offset": 0.30,
        "prong_height": 4.40,
        "prong_inset": 0.28,
        "stone_bottom_z": 11.15,
        "stone_depth": 3.35,
    }
    if fit_report is not None:
        fitted = json.loads(fit_report.read_text(encoding="utf-8"))["optimized"]["parameters"]
        parameters.update({name: float(value) for name, value in fitted.items() if name in parameters})
    hoop_major_radius = parameters["hoop_major_radius"]
    shank_radius = parameters["shank_radius"]
    stone_radius = parameters["stone_radius"]
    stone_diameter = stone_radius * 2.0

    # CadQuery's default torus lies in XY. Rotate it into XZ so side/back show
    # the hoop and front/top look down the gemstone axis.
    metal = cq.Workplane(obj=cq.Solid.makeTorus(hoop_major_radius, shank_radius)).rotate((0, 0, 0), (1, 0, 0), 90)
    shoulders = None
    for sign in (-1.0, 1.0):
        start = cq.Vector(6.0 * sign, 0.0, 8.0)
        end = cq.Vector(parameters["shoulder_end_x"] * sign, 0.0, parameters["seat_z"] + 0.3)
        support = _cylinder_between(cq, start, end, parameters["shoulder_radius"])
        shoulders = support if shoulders is None else shoulders.union(support)
    metal = metal.union(shoulders)

    seat_z = parameters["seat_z"]
    seat_outer = cq.Workplane("XY").circle(stone_radius + parameters["seat_outer_extra"]).extrude(parameters["seat_height"]).translate((0, 0, seat_z))
    seat_inner = cq.Workplane("XY").circle(max(0.9, stone_radius - parameters["seat_inner_inset"])).extrude(parameters["seat_height"] + 0.75).translate((0, 0, seat_z - 0.25))
    metal = metal.union(seat_outer.cut(seat_inner))

    prong_radius = parameters["prong_radius"]
    prong_height = parameters["prong_height"] - 0.12
    prongs = None
    prong_centers = []
    for angle in (45, 135, 225, 315):
        radians = math.radians(angle)
        radial = stone_radius + parameters["prong_offset"]
        x = radial * math.cos(radians)
        y = radial * math.sin(radians)
        top_radial = radial - parameters["prong_inset"]
        tx = top_radial * math.cos(radians)
        ty = top_radial * math.sin(radians)
        start = cq.Vector(x, y, seat_z + 0.12)
        end = cq.Vector(tx, ty, seat_z + parameters["prong_height"])
        prong = _cone_between(cq, start, end, prong_radius, prong_radius * 0.78)
        prongs = prong if prongs is None else prongs.union(prong)
        prong_centers.append([round(x, 4), round(y, 4), round(seat_z + 0.12, 4)])
    metal = metal.union(prongs)

    cavity_radius = max(0.8, stone_radius - parameters["seat_inner_inset"] + 0.16)
    cavity = cq.Workplane("XY").circle(cavity_radius).extrude(parameters["prong_height"] - 0.75).translate((0, 0, seat_z + 0.05))
    metal = metal.cut(cavity)

    stone_bottom_z = parameters["stone_bottom_z"]
    stone_depth = parameters["stone_depth"]
    pavilion_depth = stone_depth * 0.61
    pavilion = (
        cq.Workplane("XY")
        .workplane(offset=stone_bottom_z)
        .circle(stone_radius * 0.14)
        .workplane(offset=pavilion_depth)
        .circle(stone_radius)
        .loft()
    )
    crown = (
        cq.Workplane("XY")
        .workplane(offset=stone_bottom_z + pavilion_depth)
        .circle(stone_radius)
        .workplane(offset=stone_depth - pavilion_depth)
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
        "fit_report": str(fit_report) if fit_report is not None else None,
        "fitted_parameters": {name: round(value, 5) for name, value in parameters.items()},
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
    parser.add_argument("--fit-report", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.phase2_2_dir, args.phase3_dir, args.output_dir, args.fit_report), indent=2))


if __name__ == "__main__":
    main()
