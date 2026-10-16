"""Build and validate the first Bible Phase 5 exact-CAD family fixtures.

These fixtures use designed parameters to test the CAD generator itself.  They
are not reconstructed from images and do not establish metric or manufacturing
accuracy for Ring01 or STL-1.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Any

from pipeline.exact_cad_v2 import build_ring, default_family_specs


SCHEMA_VERSION = "lalitha_phase5_exact_cad_v2_benchmark_v1"


def _shape_summary(shape: Any) -> dict[str, Any]:
    return {
        "valid": bool(shape.isValid()),
        "solid_count": len(shape.Solids()),
        "volume_mm3": round(float(shape.Volume()), 6),
        "face_count": len(shape.Faces()),
        "edge_count": len(shape.Edges()),
    }


def _add_mesh(axis, shape: Any, color: str, alpha: float = 1.0) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, triangles = shape.tessellate(0.14)
    points = [(float(v.x), float(v.y), float(v.z)) for v in vertices]
    faces = [[points[index] for index in triangle] for triangle in triangles]
    mesh = Poly3DCollection(faces, linewidths=0.035, alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("#3c4148")
    axis.add_collection3d(mesh)


def _render_audit(builds: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 12), facecolor="#f4f4f4")
    for index, build in enumerate(builds, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _add_mesh(axis, build["metal"], "#d1a64b")
        for stone in build["stones"]:
            _add_mesh(axis, stone, "#5cc0df", 0.94)
        bbox = build["assembly"].BoundingBox()
        center_x = (bbox.xmin + bbox.xmax) / 2.0
        center_y = (bbox.ymin + bbox.ymax) / 2.0
        center_z = (bbox.zmin + bbox.zmax) / 2.0
        span = max(bbox.xlen, bbox.ylen, bbox.zlen) * 0.60
        axis.set_xlim(center_x - span, center_x + span)
        axis.set_ylim(center_y - span, center_y + span)
        axis.set_zlim(center_z - span, center_z + span)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=20, azim=-58)
        axis.set_axis_off()
        spec = build["spec"]
        axis.set_title(
            f"{spec.family.replace('_', ' ').upper()} · {len(spec.stones)} STONE"
            f"{'S' if len(spec.stones) != 1 else ''}",
            loc="left", fontsize=13, fontweight="bold",
        )
    figure.suptitle(
        "PHASE 5 · EXACT CAD V2 · STRUCTURAL FAMILY FIXTURES",
        fontsize=18, fontweight="bold", color="#20252b",
    )
    figure.text(
        0.5, 0.025,
        "Designed synthetic truth · exact B-rep generator validation · not image-derived CAD",
        ha="center", fontsize=11, color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.05, 0.99, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _export_family(cq, root: Path, output_dir: Path, build: dict[str, Any]) -> dict[str, str]:
    import trimesh

    family = build["spec"].family
    assembly_step = output_dir / f"{family}_assembly.step"
    metal_step = output_dir / f"{family}_metal.step"
    assembly_stl = output_dir / f"{family}_assembly.stl"
    cq.exporters.export(cq.Workplane(obj=build["assembly"]), str(assembly_step))
    cq.exporters.export(cq.Workplane(obj=build["metal"]), str(metal_step))
    # STEP remains authoritative.  OCCT's direct compound STL can contain
    # tessellation T-junctions even when every B-rep is valid, so derive a
    # closed review/print mesh independently for each exact solid.
    closed_meshes = []
    with tempfile.TemporaryDirectory(prefix="phase5_mesh_") as temp:
        temp_dir = Path(temp)
        for index, shape in enumerate([build["metal"], *build["stones"]]):
            raw_path = temp_dir / f"solid_{index:03d}.stl"
            cq.exporters.export(
                cq.Workplane(obj=shape), str(raw_path),
                tolerance=0.07, angularTolerance=0.20,
            )
            raw_mesh = trimesh.load(raw_path, force="mesh")
            pitch = 0.15 if index == 0 else max(
                0.05, min(0.12, max(raw_mesh.extents) / 55.0)
            )
            voxels = raw_mesh.voxelized(pitch=pitch).fill()
            closed = voxels.marching_cubes
            closed.apply_transform(voxels.transform)
            closed.process(validate=True)
            closed_meshes.append(closed)
    trimesh.util.concatenate(closed_meshes).export(assembly_stl)
    return {
        "assembly_step": str(assembly_step.relative_to(root)),
        "metal_step": str(metal_step.relative_to(root)),
        "assembly_stl": str(assembly_stl.relative_to(root)),
    }


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    import cadquery as cq
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    builds = [build_ring(cq, spec) for spec in default_family_specs()]
    family_reports = []
    for build in builds:
        spec = build["spec"]
        components = build["components"]
        artifacts = _export_family(cq, root, output_dir, build)
        imported = cq.importers.importStep(str(root / artifacts["assembly_step"]))
        mesh = trimesh.load(root / artifacts["assembly_stl"], force="mesh")
        expected_prongs = sum(
            stone.prong_count
            for stone in spec.stones
            if stone.setting_style == "prong"
        )
        expected_bezels = sum(
            stone.setting_style == "bezel" for stone in spec.stones
        )
        actual_prongs = sum(key.startswith("prong_") for key in components)
        actual_bezels = sum(key.startswith("bezel_") for key in components)
        component_checks = {
            "all_source_components_valid": all(
                shape.isValid() and len(shape.Solids()) == 1
                for shape in components.values()
            ),
            "stone_count_exact": len(build["stones"]) == len(spec.stones),
            "seat_count_exact": sum(key.startswith("seat_") for key in components) == len(spec.stones),
            "cavity_count_exact": sum(key.startswith("cavity_") for key in components) == len(spec.stones),
            "prong_count_exact": actual_prongs == expected_prongs,
            "bezel_count_exact": actual_bezels == expected_bezels,
            "metal_brep_valid": build["metal"].isValid(),
            "metal_single_solid": len(build["metal"].Solids()) == 1,
            "assembly_brep_valid": build["assembly"].isValid(),
            "step_roundtrip_valid": imported.val().isValid(),
            "step_roundtrip_solid_count_exact": len(imported.solids().vals()) == 1 + len(spec.stones),
            "derived_stl_watertight": bool(mesh.is_watertight),
        }
        family_reports.append({
            "object_id": spec.object_id,
            "family": spec.family,
            "designed_spec": asdict(spec),
            "component_count": len(components),
            "component_ids": sorted(components),
            "topology": {
                "stone_count": len(spec.stones),
                "seat_count": len(spec.stones),
                "cavity_count": len(spec.stones),
                "prong_count": actual_prongs,
                "bezel_count": actual_bezels,
            },
            "metal": _shape_summary(build["metal"]),
            "assembly": _shape_summary(build["assembly"]),
            "checks": component_checks,
            "passed": all(component_checks.values()),
            "artifacts": artifacts,
        })

    audit_path = output_dir / "exact_cad_v2_family_audit.png"
    _render_audit(builds, audit_path)
    passed_count = sum(item["passed"] for item in family_reports)
    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_profile": "designed_synthetic_exact_cad",
        "summary": {
            "family_count": len(family_reports),
            "passed_family_count": passed_count,
            "families": [item["family"] for item in family_reports],
            "total_stone_count": sum(item["topology"]["stone_count"] for item in family_reports),
            "total_prong_count": sum(item["topology"]["prong_count"] for item in family_reports),
            "total_bezel_count": sum(item["topology"]["bezel_count"] for item in family_reports),
        },
        "families": family_reports,
        "checks": {
            "all_family_fixtures_passed": passed_count == len(family_reports),
            "multiple_structural_families_exercised": len(family_reports) >= 4,
            "all_step_roundtrips_valid": all(
                item["checks"]["step_roundtrip_valid"] for item in family_reports
            ),
            "all_derived_meshes_watertight": all(
                item["checks"]["derived_stl_watertight"] for item in family_reports
            ),
        },
        "decision": (
            "phase5_exact_cad_generator_foundation_validated"
            if passed_count == len(family_reports)
            else "phase5_exact_cad_generator_foundation_failed"
        ),
        "artifacts": {
            "audit": str(audit_path.relative_to(root)),
        },
        "claims": {
            "designed_exact_cad_generators_validated": passed_count == len(family_reports),
            "image_derived_exact_cad_validated": False,
            "unseen_image_family_generalization_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "All fixtures use explicit designed parameters rather than measurements inferred from images.",
            "The bridge-style head is a topology fixture, not a final jewellery aesthetic.",
            "Shoulder, gallery, channel and split-shank generators remain to be implemented.",
            "No physical scale, manufacturing profile or specialist approval is available.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/phase5_exact_cad_v2_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(root, output_dir)
    print(json.dumps({
        "decision": report["decision"],
        "summary": report["summary"],
        "checks": report["checks"],
        "artifacts": report["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
