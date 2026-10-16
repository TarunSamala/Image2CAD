"""Validate Phase 4 local-edit behaviour with deterministic exact CAD solids.

This is a synthetic engineering benchmark.  It proves that the component graph
can drive local B-rep regeneration without modifying unrelated source solids.
It does not validate image-derived dimensions, Ring01 geometry, or
manufacturing accuracy.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from pipeline.component_constraint_graph import apply_local_edit


SCHEMA_VERSION = "lalitha_phase4_exact_cad_locality_v1"
OPERATIONS: dict[str, dict[str, Any]] = {
    "resize": {"diameter": 2.4, "depth": 1.4},
    "remove": {},
    "replace": {
        "semantic_class": "oval_gemstone",
        "dimensions": {"length": 2.6, "width": 2.0, "depth": 1.3},
    },
}
X_POSITIONS = (-12.0, -6.0, 0.0, 6.0, 12.0)
Y_POSITIONS = (-2.8, 2.8)
BASE_HEIGHT = 1.8


def _import_cadquery():
    try:
        import cadquery as cq
    except Exception as exc:  # pragma: no cover - depends on the CAD image
        raise RuntimeError(
            "CadQuery is unavailable. Run this benchmark in the pinned validation "
            "container after installing the required OpenGL runtime library."
        ) from exc
    return cq


def _node_lookup(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["component_id"]: node for node in graph["nodes"]}


def _stone_position(stone_index: int) -> tuple[float, float]:
    row, column = divmod(stone_index - 1, len(X_POSITIONS))
    return X_POSITIONS[column], Y_POSITIONS[row]


def _stone_dimensions(node: dict[str, Any]) -> tuple[float, float, float]:
    values = node["dimensions"]["values"]
    if node["semantic_class"] == "oval_gemstone":
        return float(values["length"]), float(values["width"]), float(values["depth"])
    diameter = float(values["diameter"])
    return diameter, diameter, float(values["depth"])


def _ellipse_prism(cq, length: float, width: float, height: float):
    return cq.Workplane("XY").ellipse(length / 2.0, width / 2.0).extrude(height).val()


def _make_setting_components(
    cq,
    graph: dict[str, Any],
    stone_index: int,
) -> dict[str, Any | None]:
    nodes = _node_lookup(graph)
    suffix = f"{stone_index:03d}"
    stone_id = f"stone_{suffix}"
    stone_node = nodes[stone_id]
    prong_start = (stone_index - 1) * 4 + 1
    component_ids = [
        stone_id,
        f"seat_{suffix}",
        f"cavity_{suffix}",
        *[f"prong_{index:03d}" for index in range(prong_start, prong_start + 4)],
    ]
    if not stone_node["active"]:
        return {component_id: None for component_id in component_ids}

    x, y = _stone_position(stone_index)
    length, width, depth = _stone_dimensions(stone_node)
    clearance = 0.09
    seat_wall = 0.28
    seat_height = 0.48
    seat_z = BASE_HEIGHT - 0.22
    stone_z = BASE_HEIGHT - 0.04

    # The stone is a faceted two-stage loft, retained as an independent solid.
    lower = cq.Workplane("XY").ellipse(length * 0.40, width * 0.40)
    stone = (
        lower
        .workplane(offset=depth * 0.38)
        .ellipse(length * 0.50, width * 0.50)
        .workplane(offset=depth * 0.62)
        .ellipse(length * 0.36, width * 0.36)
        .loft(combine=True)
        .val()
        .translate((x, y, stone_z))
    )

    outer = _ellipse_prism(
        cq, length + 2.0 * (clearance + seat_wall),
        width + 2.0 * (clearance + seat_wall), seat_height,
    )
    inner = _ellipse_prism(
        cq, length + 2.0 * clearance, width + 2.0 * clearance,
        seat_height + 0.2,
    ).translate((0.0, 0.0, -0.1))
    seat = outer.cut(inner).translate((x, y, seat_z))

    cavity = _ellipse_prism(
        cq, length + 2.0 * clearance, width + 2.0 * clearance, 1.18,
    ).translate((x, y, BASE_HEIGHT - 0.90))

    prong_radius = 0.19
    prong_height = depth + 0.52
    prong_z = BASE_HEIGHT - 0.24
    offset_x = length / 2.0 + clearance + seat_wall * 0.38
    offset_y = width / 2.0 + clearance + seat_wall * 0.38
    prongs = []
    for dx, dy in ((-offset_x, -offset_y), (offset_x, -offset_y),
                   (offset_x, offset_y), (-offset_x, offset_y)):
        prong = (
            cq.Workplane("XY")
            .circle(prong_radius)
            .extrude(prong_height)
            .faces(">Z")
            .workplane()
            .sphere(prong_radius * 1.18, combine=True)
            .val()
            .translate((x + dx, y + dy, prong_z))
        )
        prongs.append(prong)

    result: dict[str, Any | None] = {
        stone_id: stone,
        f"seat_{suffix}": seat,
        f"cavity_{suffix}": cavity,
    }
    for offset, prong in enumerate(prongs):
        result[f"prong_{prong_start + offset:03d}"] = prong
    return result


def build_source_components(cq, graph: dict[str, Any]) -> dict[str, Any | None]:
    base = (
        cq.Workplane("XY")
        .rect(32.0, 10.0)
        .extrude(BASE_HEIGHT)
        .edges("|Z")
        .fillet(0.65)
        .val()
    )
    components: dict[str, Any | None] = {"metal_body_001": base}
    for stone_index in range(1, 11):
        components.update(_make_setting_components(cq, graph, stone_index))
    return components


def _shape_hash(shape: Any | None, component_id: str, directory: Path) -> str | None:
    if shape is None:
        return None
    path = directory / f"{component_id}.brep"
    shape.exportBrep(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round_vector(vector: Any) -> list[float]:
    return [round(float(vector.x), 6), round(float(vector.y), 6), round(float(vector.z), 6)]


def component_signatures(components: dict[str, Any | None]) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="phase4_brep_") as temp:
        temp_dir = Path(temp)
        for component_id, shape in sorted(components.items()):
            if shape is None:
                signatures[component_id] = {"active": False, "brep_sha256": None}
                continue
            bbox = shape.BoundingBox()
            signatures[component_id] = {
                "active": True,
                "brep_sha256": _shape_hash(shape, component_id, temp_dir),
                "valid": bool(shape.isValid()),
                "volume_mm3": round(float(shape.Volume()), 6),
                "area_mm2": round(float(shape.Area()), 6),
                "center_mm": _round_vector(shape.Center()),
                "bounding_box_mm": [
                    round(float(bbox.xlen), 6),
                    round(float(bbox.ylen), 6),
                    round(float(bbox.zlen), 6),
                ],
                "topology": {
                    "solids": len(shape.Solids()),
                    "faces": len(shape.Faces()),
                    "edges": len(shape.Edges()),
                    "vertices": len(shape.Vertices()),
                },
            }
    return signatures


def build_derived_assembly(cq, components: dict[str, Any | None]) -> tuple[Any, Any, list[Any]]:
    base = components["metal_body_001"]
    positives = [
        shape for component_id, shape in components.items()
        if shape is not None and (component_id.startswith("seat_") or component_id.startswith("prong_"))
    ]
    cavities = [
        shape for component_id, shape in components.items()
        if shape is not None and component_id.startswith("cavity_")
    ]
    stones = [
        shape for component_id, shape in components.items()
        if shape is not None and component_id.startswith("stone_")
    ]
    metal = base.fuse(*positives).clean()
    if cavities:
        metal = metal.cut(*cavities).clean()
    assembly = cq.Compound.makeCompound([metal, *stones])
    return metal, assembly, stones


def _shape_summary(shape: Any) -> dict[str, Any]:
    return {
        "valid": bool(shape.isValid()),
        "solid_count": len(shape.Solids()),
        "volume_mm3": round(float(shape.Volume()), 6),
        "face_count": len(shape.Faces()),
    }


def _export_scenario(cq, output_dir: Path, name: str, metal: Any, assembly: Any) -> dict[str, str]:
    step_path = output_dir / f"{name}_assembly.step"
    stl_path = output_dir / f"{name}_assembly.stl"
    metal_path = output_dir / f"{name}_metal.step"
    cq.exporters.export(cq.Workplane(obj=assembly), str(step_path))
    cq.exporters.export(
        cq.Workplane(obj=assembly), str(stl_path),
        tolerance=0.08, angularTolerance=0.20,
    )
    cq.exporters.export(cq.Workplane(obj=metal), str(metal_path))
    return {
        "assembly_step": step_path.name,
        "assembly_stl": stl_path.name,
        "metal_step": metal_path.name,
    }


def _add_shape_mesh(axis, shape: Any, color: str, alpha: float = 1.0) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, triangles = shape.tessellate(0.12)
    points = [(float(v.x), float(v.y), float(v.z)) for v in vertices]
    faces = [[points[index] for index in triangle] for triangle in triangles]
    mesh = Poly3DCollection(faces, linewidths=0.04, alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("#3b424b")
    axis.add_collection3d(mesh)


def render_audit(cq, scenarios: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 9), facecolor="#f4f4f4")
    target_ids = [
        "stone_007", "seat_007", "cavity_007",
        "prong_025", "prong_026", "prong_027", "prong_028",
    ]
    for index, scenario in enumerate(scenarios, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        components = scenario["components"]
        _add_shape_mesh(axis, components["metal_body_001"], "#a9adb2", 0.10)
        for component_id in target_ids:
            shape = components[component_id]
            if shape is None:
                continue
            if component_id.startswith("stone_"):
                color, alpha = "#57bad8", 0.96
            elif component_id.startswith("cavity_"):
                color, alpha = "#d14e7b", 0.24
            else:
                color, alpha = "#d1a64b", 0.96
            _add_shape_mesh(axis, shape, color, alpha)
        axis.set_title(scenario["title"], loc="left", fontsize=13, fontweight="bold")
        # Review the local setting rather than shrinking all ten settings into
        # one panel.  The exported assemblies retain the full benchmark.
        axis.view_init(elev=62, azim=-54)
        axis.set_xlim(-9.3, -2.7)
        axis.set_ylim(-0.1, 5.7)
        axis.set_zlim(0.7, 3.7)
        axis.set_box_aspect((6.6, 5.8, 3.0))
        axis.set_axis_off()
    figure.suptitle(
        "PHASE 4 · EXACT CAD LOCALITY · STONE 007",
        fontsize=18, fontweight="bold", color="#20252b",
    )
    figure.text(
        0.5, 0.02,
        "Local setting close-up · the other nine settings remain byte-identical exact B-reps",
        ha="center", fontsize=11, color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.05, 0.99, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    cq = _import_cadquery()
    graph_path = root / "data/phase4_component_graph_v3/synthetic_ten_stone_graph.json"
    baseline_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_components = build_source_components(cq, baseline_graph)
    baseline_signatures = component_signatures(baseline_components)
    baseline_metal, baseline_assembly, baseline_stones = build_derived_assembly(
        cq, baseline_components
    )
    baseline_artifacts = _export_scenario(
        cq, output_dir, "baseline", baseline_metal, baseline_assembly
    )

    cases: list[dict[str, Any]] = []
    preview_scenarios = [{
        "title": "BASELINE · STONE 007",
        "components": baseline_components,
    }]
    representative_artifacts: dict[str, dict[str, str]] = {}
    for stone_index in range(1, 11):
        stone_id = f"stone_{stone_index:03d}"
        for operation, parameters in OPERATIONS.items():
            edited_graph, graph_audit = apply_local_edit(
                deepcopy(baseline_graph), stone_id, operation, parameters
            )
            edited_components = build_source_components(cq, edited_graph)
            edited_signatures = component_signatures(edited_components)
            expected = graph_audit["expected_affected_component_ids"]
            changed = sorted(
                component_id
                for component_id in baseline_signatures
                if baseline_signatures[component_id] != edited_signatures[component_id]
            )
            unrelated = sorted(set(baseline_signatures) - set(expected))
            metal, assembly, stones = build_derived_assembly(cq, edited_components)
            metal_summary = _shape_summary(metal)
            assembly_summary = _shape_summary(assembly)
            checks = {
                "graph_affected_set_exact": graph_audit["affected_set_exact"],
                "exact_brep_affected_set_exact": changed == expected,
                "unrelated_brep_hashes_unchanged": all(
                    baseline_signatures[item]["brep_sha256"]
                    == edited_signatures[item]["brep_sha256"]
                    for item in unrelated
                ),
                "all_active_source_solids_valid": all(
                    not signature["active"] or signature["valid"]
                    for signature in edited_signatures.values()
                ),
                "derived_metal_valid": metal_summary["valid"],
                "derived_metal_single_solid": metal_summary["solid_count"] == 1,
                "derived_assembly_valid": assembly_summary["valid"],
            }
            case = {
                "edited_component_id": stone_id,
                "operation": operation,
                "parameters": parameters,
                "expected_affected_component_ids": expected,
                "changed_exact_component_ids": changed,
                "unrelated_component_count": len(unrelated),
                "checks": checks,
                "passed": all(checks.values()),
                "derived_metal": metal_summary,
                "derived_assembly": assembly_summary,
            }
            cases.append(case)
            if stone_id == "stone_007":
                name = f"stone007_{operation}"
                representative_artifacts[operation] = _export_scenario(
                    cq, output_dir, name, metal, assembly
                )
                preview_scenarios.append({
                    "title": f"STONE 007 · {operation.upper()}",
                    "components": edited_components,
                })

    audit_path = output_dir / "exact_cad_locality_audit.png"
    render_audit(cq, preview_scenarios, audit_path)
    passed_count = sum(case["passed"] for case in cases)
    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_profile": "synthetic_exact_cad_benchmark",
        "source_graph": str(graph_path.relative_to(root)),
        "cad_kernel": "CadQuery/OCP exact B-rep",
        "units": "mm",
        "summary": {
            "case_count": len(cases),
            "passed_case_count": passed_count,
            "pass_fraction": passed_count / len(cases),
            "stone_count": 10,
            "component_count": len(baseline_signatures),
            "active_prong_count": 40,
            "operations": sorted(OPERATIONS),
        },
        "baseline": {
            "component_signatures": baseline_signatures,
            "derived_metal": _shape_summary(baseline_metal),
            "derived_assembly": _shape_summary(baseline_assembly),
        },
        "cases": cases,
        "checks": {
            "all_30_local_edits_passed": passed_count == 30,
            "all_ten_stones_exercised": len({case["edited_component_id"] for case in cases}) == 10,
            "all_three_operations_exercised": {case["operation"] for case in cases} == set(OPERATIONS),
            "all_unrelated_exact_breps_unchanged": all(
                case["checks"]["unrelated_brep_hashes_unchanged"] for case in cases
            ),
            "all_expected_exact_breps_changed": all(
                case["checks"]["exact_brep_affected_set_exact"] for case in cases
            ),
            "all_derived_metals_valid_single_solids": all(
                case["checks"]["derived_metal_valid"]
                and case["checks"]["derived_metal_single_solid"]
                for case in cases
            ),
        },
        "decision": (
            "synthetic_exact_cad_locality_validated"
            if passed_count == 30
            else "synthetic_exact_cad_locality_failed"
        ),
        "artifacts": {
            "audit": str(audit_path.relative_to(root)),
            "baseline": {
                key: str((output_dir / value).relative_to(root))
                for key, value in baseline_artifacts.items()
            },
            "stone_007": {
                operation: {
                    key: str((output_dir / value).relative_to(root))
                    for key, value in paths.items()
                }
                for operation, paths in representative_artifacts.items()
            },
        },
        "claims": {
            "synthetic_exact_cad_locality_validated": passed_count == 30,
            "ring01_image_derived_cad_validated": False,
            "image_derived_dimensions_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "This benchmark uses designed synthetic dimensions and exact component truth.",
            "It validates local B-rep regeneration and Boolean health, not image inference.",
            "The stable metal_body_001 node is the base stock; the fused metal assembly is derived and is expected to change after a local edit.",
            "Ring01 component review, metric scale and image-derived seat or cavity geometry remain unresolved.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/phase4_exact_cad_locality_v1"),
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
