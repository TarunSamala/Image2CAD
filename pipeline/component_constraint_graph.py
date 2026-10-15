"""Deterministic Phase 4 jewellery component and constraint graph API.

The graph records semantic structure, provenance, uncertainty and local edit
dependencies.  It does not infer missing component truth or manufacture CAD.
Machine proposals remain review-gated until their source observations and
cross-view identities are approved.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


SCHEMA_VERSION = "lalitha_component_constraint_graph_v1"
PROVENANCE_STATES = {"observed", "inferred", "designed", "unknown"}
REVIEW_STATES = {"pending", "approved", "needs_correction", "rejected"}
EDIT_OPERATIONS = {"resize", "remove", "replace"}
PROPAGATING_RELATIONSHIPS = {"drives_geometry"}


class ComponentGraphError(ValueError):
    """Raised when a graph or requested local edit is invalid."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def node_hashes(graph: dict[str, Any]) -> dict[str, str]:
    """Return content hashes keyed by stable component ID."""
    return {
        node["component_id"]: _canonical_hash(node)
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("component_id"), str)
    }


def validate_graph(graph: dict[str, Any]) -> list[str]:
    """Validate the portable Phase 4 graph contract without external packages."""
    issues: list[str] = []
    if graph.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version_invalid")
    for key in ("graph_id", "object_id", "authority", "units"):
        if not str(graph.get(key, "")).strip():
            issues.append(f"{key}_missing")
    if not isinstance(graph.get("revision"), int) or graph.get("revision", -1) < 0:
        issues.append("revision_invalid")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return [*issues, "nodes_missing"]
    if not isinstance(edges, list):
        return [*issues, "edges_invalid"]

    component_ids: set[str] = set()
    track_ids: set[str] = set()
    required_node_fields = {
        "component_id",
        "track_id",
        "semantic_class",
        "semantic_confidence",
        "provenance_state",
        "review_state",
        "observations",
        "parent_id",
        "support_ids",
        "pose",
        "dimensions",
        "geometry",
        "manufacturing_role",
        "dependency_rules",
        "active",
    }
    for index, node in enumerate(nodes):
        prefix = f"node_{index}"
        if not isinstance(node, dict):
            issues.append(f"{prefix}_invalid")
            continue
        missing = sorted(required_node_fields - set(node))
        issues.extend(f"{prefix}_{field}_missing" for field in missing)
        component_id = str(node.get("component_id", "")).strip()
        track_id = str(node.get("track_id", "")).strip()
        if not component_id or component_id in component_ids:
            issues.append(f"{prefix}_component_id_invalid")
        else:
            component_ids.add(component_id)
        if not track_id or track_id in track_ids:
            issues.append(f"{prefix}_track_id_invalid")
        else:
            track_ids.add(track_id)
        if not str(node.get("semantic_class", "")).strip():
            issues.append(f"{prefix}_semantic_class_missing")
        confidence = node.get("semantic_confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            issues.append(f"{prefix}_semantic_confidence_invalid")
        if node.get("provenance_state") not in PROVENANCE_STATES:
            issues.append(f"{prefix}_provenance_state_invalid")
        if node.get("review_state") not in REVIEW_STATES:
            issues.append(f"{prefix}_review_state_invalid")
        if not isinstance(node.get("observations"), dict):
            issues.append(f"{prefix}_observations_invalid")
        if not isinstance(node.get("support_ids"), list):
            issues.append(f"{prefix}_support_ids_invalid")
        for field in ("pose", "dimensions", "geometry"):
            if not isinstance(node.get(field), dict):
                issues.append(f"{prefix}_{field}_invalid")
        if not isinstance(node.get("dependency_rules"), list):
            issues.append(f"{prefix}_dependency_rules_invalid")
        if not isinstance(node.get("active"), bool):
            issues.append(f"{prefix}_active_invalid")

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        parent = node.get("parent_id")
        if parent is not None and parent not in component_ids:
            issues.append(f"node_{index}_parent_missing")
        for support_id in node.get("support_ids", []):
            if support_id not in component_ids:
                issues.append(f"node_{index}_support_missing")

    dependency_adjacency = {component_id: [] for component_id in component_ids}
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        prefix = f"edge_{index}"
        if not isinstance(edge, dict):
            issues.append(f"{prefix}_invalid")
            continue
        edge_id = str(edge.get("edge_id", "")).strip()
        if not edge_id or edge_id in edge_ids:
            issues.append(f"{prefix}_id_invalid")
        else:
            edge_ids.add(edge_id)
        source = edge.get("source")
        target = edge.get("target")
        if source not in component_ids or target not in component_ids:
            issues.append(f"{prefix}_endpoint_invalid")
        relationship = str(edge.get("relationship", "")).strip()
        if not relationship:
            issues.append(f"{prefix}_relationship_missing")
        if edge.get("provenance_state") not in PROVENANCE_STATES:
            issues.append(f"{prefix}_provenance_state_invalid")
        confidence = edge.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            issues.append(f"{prefix}_confidence_invalid")
        if (
            relationship in PROPAGATING_RELATIONSHIPS
            and source in component_ids
            and target in component_ids
        ):
            dependency_adjacency[source].append(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(component_id: str) -> bool:
        if component_id in visiting:
            return True
        if component_id in visited:
            return False
        visiting.add(component_id)
        if any(visit(target) for target in dependency_adjacency[component_id]):
            return True
        visiting.remove(component_id)
        visited.add(component_id)
        return False

    if any(visit(component_id) for component_id in sorted(component_ids)):
        issues.append("dependency_cycle_detected")
    return sorted(set(issues))


def dependency_closure(graph: dict[str, Any], component_id: str) -> list[str]:
    """Return the edited component plus every transitively dependent component."""
    component_ids = {node["component_id"] for node in graph.get("nodes", [])}
    if component_id not in component_ids:
        raise ComponentGraphError(f"Unknown component: {component_id}")
    adjacency = {item: [] for item in component_ids}
    for edge in graph.get("edges", []):
        if edge.get("relationship") in PROPAGATING_RELATIONSHIPS:
            adjacency[edge["source"]].append(edge["target"])
    closure: set[str] = set()
    pending = [component_id]
    while pending:
        current = pending.pop()
        if current in closure:
            continue
        closure.add(current)
        pending.extend(adjacency[current])
    return sorted(closure)


def apply_local_edit(
    graph: dict[str, Any],
    component_id: str,
    operation: str,
    parameters: dict[str, Any] | None = None,
    transaction_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply an auditable graph-state edit without regenerating unrelated nodes."""
    issues = validate_graph(graph)
    if issues:
        raise ComponentGraphError(f"Graph is invalid: {', '.join(issues)}")
    if operation not in EDIT_OPERATIONS:
        raise ComponentGraphError(f"Unsupported edit operation: {operation}")
    parameters = deepcopy(parameters or {})
    transaction_id = transaction_id or (
        f"tx_{graph['revision'] + 1:04d}_{component_id}_{operation}"
    )
    before_ids = [node["component_id"] for node in graph["nodes"]]
    before_hashes = node_hashes(graph)
    expected = dependency_closure(graph, component_id)
    edited = deepcopy(graph)
    edited["revision"] += 1
    node_lookup = {node["component_id"]: node for node in edited["nodes"]}
    root = node_lookup[component_id]
    root["last_edit"] = {
        "transaction_id": transaction_id,
        "operation": operation,
        "parameters": parameters,
    }
    if operation == "remove":
        root["active"] = False
    elif operation == "resize":
        values = root["dimensions"].setdefault("values", {})
        values.update(parameters)
    elif operation == "replace":
        if "semantic_class" in parameters:
            root["semantic_class"] = str(parameters["semantic_class"])
        replacement_dimensions = parameters.get("dimensions")
        if isinstance(replacement_dimensions, dict):
            root["dimensions"].setdefault("values", {}).update(
                replacement_dimensions
            )
    for dependent_id in expected:
        if dependent_id == component_id:
            continue
        dependent = node_lookup[dependent_id]
        dependent["dependency_state"] = "regeneration_required"
        dependent["upstream_edit"] = {
            "transaction_id": transaction_id,
            "source_component_id": component_id,
            "operation": operation,
        }
    edited.setdefault("edit_transactions", []).append(
        {
            "transaction_id": transaction_id,
            "component_id": component_id,
            "operation": operation,
            "parameters": parameters,
            "affected_component_ids": expected,
        }
    )
    after_ids = [node["component_id"] for node in edited["nodes"]]
    after_hashes = node_hashes(edited)
    changed = sorted(
        component_id
        for component_id in before_hashes
        if before_hashes[component_id] != after_hashes[component_id]
    )
    unrelated = sorted(set(before_ids) - set(expected))
    audit = {
        "transaction_id": transaction_id,
        "operation": operation,
        "edited_component_id": component_id,
        "expected_affected_component_ids": expected,
        "changed_component_ids": changed,
        "unrelated_component_ids": unrelated,
        "stable_component_ids": before_ids == after_ids,
        "affected_set_exact": changed == expected,
        "unrelated_geometry_state_unchanged": all(
            before_hashes[item] == after_hashes[item] for item in unrelated
        ),
        "graph_valid_after_edit": not validate_graph(edited),
        "note": "Graph-state dependency test only; no CAD geometry was generated.",
    }
    return edited, audit


def make_node(
    component_id: str,
    semantic_class: str,
    *,
    parent_id: str | None,
    provenance_state: str,
    review_state: str,
    semantic_confidence: float,
    manufacturing_role: str,
    observations: dict[str, Any] | None = None,
    support_ids: list[str] | None = None,
    dimensions: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a complete node with explicit unknowns rather than omitted fields."""
    return {
        "component_id": component_id,
        "track_id": f"track_{component_id}",
        "semantic_class": semantic_class,
        "semantic_confidence": semantic_confidence,
        "provenance_state": provenance_state,
        "review_state": review_state,
        "observations": observations or {},
        "parent_id": parent_id,
        "support_ids": support_ids or [],
        "pose": {
            "coordinate_frame": "object_normalized",
            "translation": None,
            "rotation_quaternion_xyzw": None,
            "uncertainty": "unknown",
        },
        "dimensions": dimensions
        or {"units": "normalized", "values": {}, "uncertainty": {}},
        "geometry": geometry
        or {"generator": None, "freeform_surface": None, "parameters": {}},
        "manufacturing_role": manufacturing_role,
        "dependency_rules": [],
        "active": True,
    }


def make_edge(
    source: str,
    target: str,
    relationship: str,
    *,
    provenance_state: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "edge_id": f"edge_{source}_{relationship}_{target}",
        "source": source,
        "target": target,
        "relationship": relationship,
        "provenance_state": provenance_state,
        "confidence": confidence,
    }
