"""Build the first Bible Phase 4 graph contract and synthetic edit benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from pipeline.component_constraint_graph import (
    SCHEMA_VERSION,
    apply_local_edit,
    make_edge,
    make_node,
    validate_graph,
)


VIEWS = ("front", "side", "top", "angled", "back")


def _base_graph(graph_id: str, object_id: str, authority: str, units: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_id": graph_id,
        "object_id": object_id,
        "authority": authority,
        "units": units,
        "revision": 0,
        "nodes": [],
        "edges": [],
        "edit_transactions": [],
    }


def _ring01_observations(
    manifest: dict[str, Any], label: str, *, aggregate_instance_mask: bool = False
) -> dict[str, Any]:
    observations = {}
    for view in VIEWS:
        proposal = manifest["views"][view]["proposals"][label]
        observations[view] = {
            "visible_mask": proposal["path"],
            "amodal_mask": None,
            "sha256": proposal["sha256"],
            "confidence": None,
            "provenance_state": "inferred",
            "review_state": manifest["views"][view]["reviews"][label]["decision"],
            "aggregate_instance_mask": aggregate_instance_mask,
        }
    return observations


def build_ring01_proposal(manifest: dict[str, Any]) -> dict[str, Any]:
    graph = _base_graph(
        "ring01_component_graph_proposal_v1",
        "ring01",
        "machine_proposal_review_required",
        "normalized",
    )
    nodes = [
        make_node(
            "jewellery_001",
            "ring",
            parent_id=None,
            provenance_state="observed",
            review_state="pending",
            semantic_confidence=1.0,
            manufacturing_role="assembly_root",
            observations=_ring01_observations(manifest, "jewelry"),
        )
    ]
    class_to_label = {
        "shank": "shank",
        "setting": "setting",
        "round_gemstone": "stone_visible",
        "prong": "prongs",
    }
    parent = {
        "shank": "jewellery_001",
        "setting": "jewellery_001",
        "round_gemstone": "setting_001",
        "prong": "setting_001",
    }
    role = {
        "shank": "primary_metal_body",
        "setting": "stone_support",
        "round_gemstone": "separate_gemstone",
        "prong": "stone_retention",
    }
    for item in manifest["component_catalog"]:
        semantic_class = item["class"]
        nodes.append(
            make_node(
                item["component_id"],
                semantic_class,
                parent_id=parent[semantic_class],
                provenance_state="inferred",
                review_state=item["review"]["decision"],
                semantic_confidence=0.5,
                manufacturing_role=role[semantic_class],
                observations=_ring01_observations(
                    manifest,
                    class_to_label[semantic_class],
                    aggregate_instance_mask=semantic_class == "prong",
                ),
                support_ids=["setting_001"]
                if semantic_class in {"round_gemstone", "prong"}
                else [],
            )
        )
    for component_id, semantic_class, role_name in (
        ("seat_001", "stone_seat", "stone_location"),
        ("cavity_001", "stone_cavity", "negative_geometry"),
    ):
        nodes.append(
            make_node(
                component_id,
                semantic_class,
                parent_id="setting_001",
                provenance_state="unknown",
                review_state="pending",
                semantic_confidence=0.0,
                manufacturing_role=role_name,
                support_ids=["setting_001"],
            )
        )
    graph["nodes"] = nodes
    edges = [
        make_edge(
            "jewellery_001", "shank_001", "contains",
            provenance_state="inferred", confidence=0.5,
        ),
        make_edge(
            "jewellery_001", "setting_001", "contains",
            provenance_state="inferred", confidence=0.5,
        ),
        make_edge(
            "setting_001", "stone_001", "supports",
            provenance_state="inferred", confidence=0.5,
        ),
        make_edge(
            "seat_001", "stone_001", "locates",
            provenance_state="unknown", confidence=0.0,
        ),
        make_edge(
            "cavity_001", "stone_001", "receives",
            provenance_state="unknown", confidence=0.0,
        ),
    ]
    for dependent in ["seat_001", "cavity_001", *[f"prong_{i:03d}" for i in range(1, 5)]]:
        edges.append(
            make_edge(
                "stone_001", dependent, "drives_geometry",
                provenance_state="unknown", confidence=0.0,
            )
        )
    for prong in [f"prong_{i:03d}" for i in range(1, 5)]:
        edges.append(
            make_edge(
                prong, "stone_001", "retains",
                provenance_state="inferred", confidence=0.5,
            )
        )
    graph["edges"] = edges
    graph["gate"] = {
        "reviewed_component_graph": False,
        "individual_prong_masks_available": False,
        "seat_and_cavity_observed": False,
        "geometry_generation_authorized": False,
        "reason": "All component identities remain pending and prongs use one aggregate mask.",
    }
    return graph


def build_synthetic_ten_stone_graph() -> dict[str, Any]:
    graph = _base_graph(
        "synthetic_ten_stone_graph_v1",
        "synthetic_ten_stone_ring",
        "synthetic_exact_graph_truth",
        "mm",
    )
    graph["nodes"].extend(
        [
            make_node(
                "jewellery_001", "ring", parent_id=None,
                provenance_state="designed", review_state="approved",
                semantic_confidence=1.0, manufacturing_role="assembly_root",
            ),
            make_node(
                "metal_body_001", "metal_body", parent_id="jewellery_001",
                provenance_state="designed", review_state="approved",
                semantic_confidence=1.0, manufacturing_role="primary_metal_body",
            ),
        ]
    )
    graph["edges"].append(
        make_edge(
            "jewellery_001", "metal_body_001", "contains",
            provenance_state="designed", confidence=1.0,
        )
    )
    prong_index = 1
    for stone_index in range(1, 11):
        suffix = f"{stone_index:03d}"
        stone_id = f"stone_{suffix}"
        seat_id = f"seat_{suffix}"
        cavity_id = f"cavity_{suffix}"
        graph["nodes"].append(
            make_node(
                stone_id, "round_gemstone", parent_id="jewellery_001",
                provenance_state="designed", review_state="approved",
                semantic_confidence=1.0, manufacturing_role="separate_gemstone",
                support_ids=[seat_id],
                dimensions={
                    "units": "mm",
                    "values": {"diameter": 2.0, "depth": 1.2},
                    "uncertainty": {"diameter": 0.0, "depth": 0.0},
                },
            )
        )
        for component_id, semantic_class, role_name in (
            (seat_id, "stone_seat", "stone_location"),
            (cavity_id, "stone_cavity", "negative_geometry"),
        ):
            graph["nodes"].append(
                make_node(
                    component_id, semantic_class, parent_id="metal_body_001",
                    provenance_state="designed", review_state="approved",
                    semantic_confidence=1.0, manufacturing_role=role_name,
                    support_ids=["metal_body_001"],
                )
            )
            graph["edges"].append(
                make_edge(
                    stone_id, component_id, "drives_geometry",
                    provenance_state="designed", confidence=1.0,
                )
            )
        graph["edges"].extend(
            [
                make_edge(
                    seat_id, stone_id, "locates",
                    provenance_state="designed", confidence=1.0,
                ),
                make_edge(
                    cavity_id, stone_id, "receives",
                    provenance_state="designed", confidence=1.0,
                ),
            ]
        )
        for _ in range(4):
            prong_id = f"prong_{prong_index:03d}"
            prong_index += 1
            graph["nodes"].append(
                make_node(
                    prong_id, "prong", parent_id="metal_body_001",
                    provenance_state="designed", review_state="approved",
                    semantic_confidence=1.0, manufacturing_role="stone_retention",
                    support_ids=["metal_body_001"],
                )
            )
            graph["edges"].extend(
                [
                    make_edge(
                        stone_id, prong_id, "drives_geometry",
                        provenance_state="designed", confidence=1.0,
                    ),
                    make_edge(
                        prong_id, stone_id, "retains",
                        provenance_state="designed", confidence=1.0,
                    ),
                ]
            )
    return graph


def benchmark_ten_stone_edits(graph: dict[str, Any]) -> dict[str, Any]:
    cases = []
    operations = {
        "resize": {"diameter": 2.4, "depth": 1.4},
        "remove": {},
        "replace": {
            "semantic_class": "oval_gemstone",
            "dimensions": {"length": 2.6, "width": 2.0, "depth": 1.3},
        },
    }
    for stone_index in range(1, 11):
        stone_id = f"stone_{stone_index:03d}"
        for operation, parameters in operations.items():
            _, audit = apply_local_edit(
                deepcopy(graph), stone_id, operation, parameters
            )
            passed = all(
                audit[key]
                for key in (
                    "stable_component_ids",
                    "affected_set_exact",
                    "unrelated_geometry_state_unchanged",
                    "graph_valid_after_edit",
                )
            )
            cases.append({**audit, "passed": passed})
    passed = sum(case["passed"] for case in cases)
    return {
        "case_count": len(cases),
        "passed_case_count": passed,
        "pass_fraction": passed / len(cases),
        "all_stone_ids_exercised": len(
            {case["edited_component_id"] for case in cases}
        ) == 10,
        "operations": sorted(operations),
        "cases": cases,
    }


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (root / "data/ring01_ground_truth_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ring01 = build_ring01_proposal(manifest)
    synthetic = build_synthetic_ten_stone_graph()
    ring01_issues = validate_graph(ring01)
    synthetic_issues = validate_graph(synthetic)
    benchmark = benchmark_ten_stone_edits(synthetic)
    output_dir.mkdir(parents=True, exist_ok=True)
    ring01_path = output_dir / "ring01_component_graph.json"
    synthetic_path = output_dir / "synthetic_ten_stone_graph.json"
    ring01_path.write_text(json.dumps(ring01, indent=2) + "\n", encoding="utf-8")
    synthetic_path.write_text(
        json.dumps(synthetic, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "lalitha_phase4_component_graph_report_v1",
        "execution_profile": "images_only",
        "ring01": {
            "node_count": len(ring01["nodes"]),
            "edge_count": len(ring01["edges"]),
            "schema_issues": ring01_issues,
            "reviewed_component_count": sum(
                node["review_state"] == "approved" for node in ring01["nodes"]
            ),
            "pending_component_count": sum(
                node["review_state"] != "approved" for node in ring01["nodes"]
            ),
            "geometry_generation_authorized": False,
            "artifact": str(ring01_path.relative_to(root)),
        },
        "synthetic_ten_stone_edit_benchmark": benchmark,
        "checks": {
            "ring01_graph_schema_valid": not ring01_issues,
            "synthetic_graph_schema_valid": not synthetic_issues,
            "synthetic_ten_stone_edit_test_passed": benchmark["pass_fraction"] == 1.0,
            "no_unrelated_graph_state_changes": all(
                case["unrelated_geometry_state_unchanged"]
                for case in benchmark["cases"]
            ),
            "stable_ids_across_all_edits": all(
                case["stable_component_ids"] for case in benchmark["cases"]
            ),
            "reviewed_ring01_graph_available": False,
            "phase4_exit_gate_passed": False,
        },
        "decision": "graph_api_and_synthetic_contract_implemented_reviewed_evidence_blocked",
        "artifacts": {
            "ring01_component_graph": str(ring01_path.relative_to(root)),
            "synthetic_ten_stone_graph": str(synthetic_path.relative_to(root)),
        },
        "claims": {
            "ring01_component_identity_validated": False,
            "individual_prong_segmentation_validated": False,
            "seat_or_cavity_geometry_validated": False,
            "cad_locality_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The ten-stone benchmark validates graph-state locality, not CAD Boolean geometry.",
            "Ring01 component identities and masks remain pending human review.",
            "The four Ring01 prongs currently share an aggregate prong mask.",
            "Seat and cavity nodes are required placeholders with unknown image evidence.",
            "Image-only evidence does not establish absolute physical scale.",
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
        "--output-dir",
        type=Path,
        default=Path("data/phase4_component_graph_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(root, output_dir)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "ring01": report["ring01"],
                "benchmark": {
                    key: value
                    for key, value in report["synthetic_ten_stone_edit_benchmark"].items()
                    if key != "cases"
                },
                "checks": report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
