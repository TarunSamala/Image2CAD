"""Build a valid, editable Ring01 solitaire baseline from metadata.

The output keeps the metal and gemstone separate.  The metal contains a
boolean stone seat/cavity, allowing downstream manufacturing checks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build(metadata_path: Path, output_dir: Path) -> dict[str, str]:
    import cadquery as cq

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stone = metadata["vision"]["stones"][0]
    diameter = float(stone.get("diameter_mm") or 6.5)
    r = diameter / 2.0

    # Ring axis is Z.  A torus gives a smooth circular shank without relying
    # on the generative model's coordinate conventions.
    metal = cq.Workplane(obj=cq.Solid.makeTorus(10.0, 1.35))
    # Shoulders connect the shank to the elevated head.  Without these, the
    # seat can look correct while exporting as a disconnected second solid.
    shoulders = None
    for x in (-1.0, 1.0):
        # Sloped shoulder from the shank tube to the underside of the head.
        start = cq.Vector(9.0 * x, 0, 0)
        end = cq.Vector(3.6 * x, 0, 10.8)
        direction = end.sub(start)
        support = cq.Workplane(obj=cq.Solid.makeCylinder(1.0, direction.Length, start, direction.normalized()))
        shoulders = support if shoulders is None else shoulders.union(support)
    metal = metal.union(shoulders)
    # Raised shoulder/seat beneath the gemstone.
    seat_outer = cq.Workplane("XY").circle(r + 1.2).extrude(1.2).translate((0, 0, 10.7))
    seat_inner = cq.Workplane("XY").circle(max(0.8, r - 0.35)).extrude(2.0).translate((0, 0, 10.4))
    seat = seat_outer.cut(seat_inner)
    metal = metal.union(seat)

    # Four prongs are deliberately separate parametric features before fuse.
    prong_radius = 0.42
    prongs = None
    for angle in (45, 135, 225, 315):
        a = math.radians(angle)
        x, y = (r + 0.30) * math.cos(a), (r + 0.30) * math.sin(a)
        p = cq.Workplane("XY").circle(prong_radius).extrude(4.6).translate((x, y, 11.0))
        prongs = p if prongs is None else prongs.union(p)
    metal = metal.union(prongs)

    # Hidden clearance below the stone: this is the cavity claim represented
    # in the artifact, and is independently inspectable in the STEP model.
    cavity = cq.Workplane("XY").circle(r - 0.18).extrude(3.6).translate((0, 0, 11.1))
    metal = metal.cut(cavity)

    # A simple round brilliant proxy: pavilion + girdle + crown.
    pavilion = cq.Workplane("XY").workplane(offset=11.3).circle(r * 0.58).workplane(offset=2.0).circle(r).loft()
    crown = cq.Workplane("XY").workplane(offset=13.3).circle(r).workplane(offset=1.35).circle(r * 0.55).loft()
    gemstone = pavilion.union(crown)

    metal_path = output_dir / "ring01_metal.step"
    stone_path = output_dir / "ring01_stone.step"
    assembly_path = output_dir / "ring01_parametric.step"
    stl_path = output_dir / "ring01_parametric.stl"
    cq.exporters.export(metal, str(metal_path))
    cq.exporters.export(gemstone, str(stone_path))
    cq.exporters.export(cq.Compound.makeCompound([metal.val(), gemstone.val()]), str(assembly_path))
    cq.exporters.export(cq.Compound.makeCompound([metal.val(), gemstone.val()]), str(stl_path))
    result = {
        "metal": str(metal_path), "stone": str(stone_path),
        "assembly": str(assembly_path), "stl": str(stl_path),
        "cavity": "metal boolean cut below stone", "prongs": "4",
        "validation": {
            "metal_solid_count": len(metal.solids().vals()),
            "stone_solid_count": len(gemstone.solids().vals()),
            "assembly_solid_count": len(cq.Compound.makeCompound([metal.val(), gemstone.val()]).Solids()),
            "metal_is_single_solid": len(metal.solids().vals()) == 1,
            "stone_is_single_solid": len(gemstone.solids().vals()) == 1,
        },
    }
    (output_dir / "ring01_build.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/ring01_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ring01_parametric"))
    args = parser.parse_args()
    print(json.dumps(build(args.metadata, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
