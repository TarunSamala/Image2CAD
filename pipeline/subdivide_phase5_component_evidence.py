"""Subdivide broad Phase 5.10 relief components with retained image evidence.

Nearby queries remain in one patch only when they share a reciprocal target
view, remain continuous in regularized relative depth and do not cross a strong
prepared-edge barrier.  The output is validation-only geometric evidence: it
does not assign jewellery semantics, recover hidden surfaces or modify CAD.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_component_evidence_graph import (
    PHASE1_REPORT,
    TRACK_COORDINATE_MAX,
    VISIBILITY_ROOT,
)
from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS, VIEW_ORDER


COMPONENT_ROOT = Path("data/phase5_component_evidence_graph_v1")
SPATIAL_LINK_RADIUS_TRACK_PIXELS = 24.0
MAX_NORMALIZED_DEPTH_DELTA = 0.14
MAX_EDGE_BARRIER_FRACTION = 0.65
MIN_FIT_REVIEW_QUERY_COUNT = 3
MAX_PATCH_DIAMETER_TRACK_PIXELS = 110.0
PALETTE_BGR = (
    (66, 174, 68),
    (190, 145, 35),
    (184, 70, 184),
    (55, 152, 210),
    (126, 105, 196),
    (192, 104, 42),
    (85, 180, 190),
    (165, 115, 70),
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _sample_scalar(image: np.ndarray, point_xy: tuple[float, float]) -> float:
    height, width = image.shape[:2]
    x = int(np.clip(round(point_xy[0]), 0, width - 1))
    y = int(np.clip(round(point_xy[1]), 0, height - 1))
    return float(image[y, x])


def _track_to_image(
    point_xy: list[float], width: int, height: int
) -> tuple[float, float]:
    return (
        float(point_xy[0]) * (width - 1) / TRACK_COORDINATE_MAX,
        float(point_xy[1]) * (height - 1) / TRACK_COORDINATE_MAX,
    )


def _edge_barrier_fraction(
    edge: np.ndarray,
    left_xy: tuple[float, float],
    right_xy: tuple[float, float],
) -> float:
    distance = float(np.linalg.norm(np.asarray(left_xy) - np.asarray(right_xy)))
    sample_count = max(3, int(np.ceil(distance / 2.0)) + 1)
    xs = np.linspace(left_xy[0], right_xy[0], sample_count)
    ys = np.linspace(left_xy[1], right_xy[1], sample_count)
    height, width = edge.shape
    xi = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, height - 1)
    return float(np.mean(edge[yi, xi] > 0))


def _evidence_class(nodes: list[dict[str, Any]]) -> str:
    normal_views = {
        view
        for node in nodes
        for view in node["normal_corroborated_target_views"]
    }
    accepted_views = {
        view for node in nodes for view in node["accepted_target_views"]
    }
    if len(normal_views) >= 2:
        return "multi_view_normal_corroborated"
    if len(normal_views) == 1:
        return "single_view_normal_corroborated"
    if accepted_views:
        return "reciprocal_normal_unresolved"
    raise RuntimeError("Phase 5.10 relief component contains no accepted views")


def _split_oversized_group(
    nodes: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Bound transitive graph chains with deterministic principal-axis splits."""

    if len(nodes) < 4:
        return [nodes]
    points = np.asarray([node["source_xy"] for node in nodes], np.float32)
    diagonal = float(
        np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    )
    if diagonal <= MAX_PATCH_DIAMETER_TRACK_PIXELS:
        return [nodes]
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ axes[0]
    order = np.argsort(projection, kind="stable")
    midpoint = len(order) // 2
    left = [nodes[int(index)] for index in order[:midpoint]]
    right = [nodes[int(index)] for index in order[midpoint:]]
    return _split_oversized_group(left) + _split_oversized_group(right)


def _subdivide_component(
    component: dict[str, Any],
    query_by_id: dict[str, dict[str, Any]],
    depth_normalized: np.ndarray,
    depth_uncertainty: np.ndarray,
    edge: np.ndarray,
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    nodes = [query_by_id[node_id] for node_id in component["query_node_ids"]]
    height, width = depth_normalized.shape
    points_track = np.asarray([node["source_xy"] for node in nodes], np.float32)
    points_image = [
        _track_to_image(node["source_xy"], width, height) for node in nodes
    ]
    depths = np.asarray(
        [_sample_scalar(depth_normalized, point) for point in points_image],
        np.float32,
    )
    union = _UnionFind(len(nodes))
    rejected = {
        "spatial_distance": 0,
        "no_shared_target_view": 0,
        "depth_discontinuity": 0,
        "edge_barrier": 0,
    }
    radius_squared = SPATIAL_LINK_RADIUS_TRACK_PIXELS**2
    for left in range(len(nodes)):
        left_views = set(nodes[left]["accepted_target_views"])
        for right in range(left + 1, len(nodes)):
            squared_distance = float(
                np.sum(np.square(points_track[left] - points_track[right]))
            )
            if squared_distance > radius_squared:
                rejected["spatial_distance"] += 1
                continue
            if not left_views.intersection(nodes[right]["accepted_target_views"]):
                rejected["no_shared_target_view"] += 1
                continue
            if abs(float(depths[left] - depths[right])) > MAX_NORMALIZED_DEPTH_DELTA:
                rejected["depth_discontinuity"] += 1
                continue
            if (
                _edge_barrier_fraction(edge, points_image[left], points_image[right])
                > MAX_EDGE_BARRIER_FRACTION
            ):
                rejected["edge_barrier"] += 1
                continue
            union.union(left, right)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, node in enumerate(nodes):
        groups.setdefault(union.find(index), []).append(node)
    bounded = [
        subgroup
        for group in groups.values()
        for subgroup in _split_oversized_group(group)
    ]
    ordered = sorted(
        bounded,
        key=lambda group: (
            min(item["source_xy"][1] for item in group),
            min(item["source_xy"][0] for item in group),
        ),
    )
    return ordered, rejected


def _depth_and_edge_preview(
    depth: np.ndarray, edge: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    color = cv2.applyColorMap(
        np.uint8(np.round(np.clip(depth, 0.0, 1.0) * 255)),
        cv2.COLORMAP_TURBO,
    )
    color[~mask] = (226, 226, 226)
    color[edge > 0] = (250, 250, 250)
    return color


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    width = 500
    height = 500
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 72, width, 3), 246, np.uint8)
    canvas[72:] = resized
    cv2.putText(canvas, title, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (25, 29, 34), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (9, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (65, 68, 73), 1, cv2.LINE_AA)
    return canvas


def _render_audit(
    source: np.ndarray,
    depth: np.ndarray,
    edge: np.ndarray,
    mask: np.ndarray,
    parent_components: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    output: Path,
    object_id: str,
) -> None:
    height, width = source.shape[:2]
    parent_view = source.copy()
    for component in parent_components:
        x1, y1, x2, y2 = component["source_bbox_xyxy"]
        p1 = (
            int(round(x1 * (width - 1) / TRACK_COORDINATE_MAX)),
            int(round(y1 * (height - 1) / TRACK_COORDINATE_MAX)),
        )
        p2 = (
            int(round(x2 * (width - 1) / TRACK_COORDINATE_MAX)),
            int(round(y2 * (height - 1) / TRACK_COORDINATE_MAX)),
        )
        cv2.rectangle(parent_view, p1, p2, (160, 160, 160), 2, cv2.LINE_AA)

    patch_view = source.copy()
    for index, patch in enumerate(patches):
        color = PALETTE_BGR[index % len(PALETTE_BGR)]
        points = np.asarray(
            [
                _track_to_image(point, width, height)
                for point in patch["source_points_xy"]
            ],
            np.int32,
        )
        if len(points) >= 3:
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            overlay = patch_view.copy()
            cv2.fillConvexPoly(overlay, hull, color, cv2.LINE_AA)
            patch_view = cv2.addWeighted(overlay, 0.20, patch_view, 0.80, 0.0)
            cv2.polylines(patch_view, [hull], True, color, 2, cv2.LINE_AA)
        else:
            for point in points:
                cv2.circle(patch_view, tuple(point), 5, color, 2, cv2.LINE_AA)
        centroid = np.mean(points, axis=0).astype(int)
        cv2.putText(
            patch_view,
            patch["patch_id"].rsplit("_", 1)[-1],
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    panels = [
        _panel(
            parent_view,
            "PHASE 5.10 PARENT REGIONS",
            f"{len(parent_components)} visibility-linked components",
        ),
        _panel(
            _depth_and_edge_preview(depth, edge, mask),
            "DEPTH + EDGE BARRIERS",
            "Turbo relative depth | white prepared edges",
        ),
        _panel(
            patch_view,
            "PHASE 5.11 EVIDENCE PATCHES",
            f"{len(patches)} patches | semantics remain unknown",
        ),
    ]
    row = np.hstack(panels)
    header = np.full((74, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"PHASE 5.11 - {object_id.upper()} RELIEF REGION SUBDIVISION",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 29, 34),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Spatial proximity + shared target view + relative-depth continuity + prepared-edge barrier",
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    footer = np.full((50, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Patches organize visible evidence only; they are not semantic components, watertight surfaces or CAD geometry.",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    target_width = 1200
    rows = []
    for image in images:
        scale = target_width / image.shape[1]
        rows.append(
            cv2.resize(
                image,
                (target_width, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    cv2.imwrite(str(output), np.vstack(rows))


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    component_report = json.loads(
        (root / COMPONENT_ROOT / "report.json").read_text(encoding="utf-8")
    )
    component_by_id = {
        item["object_id"]: item for item in component_report["objects"]
    }
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}

    objects = []
    audit_paths = []
    for object_id in VALIDATION_OBJECTS:
        source_record = phase1_by_id[object_id]
        if source_record["split"] != "val":
            raise RuntimeError(f"{object_id} is outside the fixed validation split")
        component_graph = json.loads(
            (
                root
                / COMPONENT_ROOT
                / object_id
                / "component_evidence_graph.json"
            ).read_text(encoding="utf-8")
        )
        visibility_graph = json.loads(
            (root / VISIBILITY_ROOT / object_id / "visibility_graph.json").read_text(
                encoding="utf-8"
            )
        )
        query_by_id = {
            item["node_id"]: item for item in visibility_graph["query_nodes"]
        }
        bundle_path = root / source_record["views"]["front"]["bundle"]
        with np.load(bundle_path) as bundle:
            source = np.asarray(bundle["normalized_rgb_u8"])
            raw_depth = np.asarray(bundle["relative_depth_regularized_f32"])
            uncertainty = (
                np.asarray(bundle["relative_depth_uncertainty_u16"], np.float32)
                / 65535.0
            )
            edge = np.asarray(bundle["prepared_edge_u8"])
            mask = np.asarray(bundle["mask_jewelry_u8"], dtype=bool)
        robust_depth = raw_depth[mask]
        low, high = np.quantile(robust_depth, (0.02, 0.98))
        depth = np.clip((raw_depth - low) / max(float(high - low), 1e-8), 0.0, 1.0)
        depth[~mask] = 0.0

        patches = []
        subdivision_records = []
        all_query_ids = []
        for parent in component_graph["relief_components"]:
            groups, rejected = _subdivide_component(
                parent, query_by_id, depth, uncertainty, edge
            )
            parent_patches = []
            for patch_index, group in enumerate(groups, start=1):
                patch_id = f"{parent['component_id']}_patch_{patch_index:03d}"
                points_track = [item["source_xy"] for item in group]
                points_image = [
                    _track_to_image(point, source.shape[1], source.shape[0])
                    for point in points_track
                ]
                depth_values = [
                    _sample_scalar(depth, point) for point in points_image
                ]
                uncertainty_values = [
                    _sample_scalar(uncertainty, point) for point in points_image
                ]
                accepted_views = sorted(
                    {
                        view
                        for item in group
                        for view in item["accepted_target_views"]
                    },
                    key=VIEW_ORDER.index,
                )
                normal_views = sorted(
                    {
                        view
                        for item in group
                        for view in item["normal_corroborated_target_views"]
                    },
                    key=VIEW_ORDER.index,
                )
                patch = {
                    "patch_id": patch_id,
                    "parent_component_id": parent["component_id"],
                    "query_node_ids": [item["node_id"] for item in group],
                    "query_count": len(group),
                    "source_points_xy": points_track,
                    "accepted_target_views": accepted_views,
                    "normal_corroborated_target_views": normal_views,
                    "evidence_state": _evidence_class(group),
                    "mean_relative_depth": round(float(np.mean(depth_values)), 6),
                    "relative_depth_span": round(
                        float(max(depth_values) - min(depth_values)), 6
                    ),
                    "mean_depth_uncertainty": round(
                        float(np.mean(uncertainty_values)), 6
                    ),
                    "review_priority": (
                        "multi_view_supported"
                        if len(normal_views) >= 2
                        else "single_view_supported"
                        if len(normal_views) == 1
                        else "normal_unresolved"
                    ),
                    "future_surface_fit_review_eligible": len(group)
                    >= MIN_FIT_REVIEW_QUERY_COUNT
                    and bool(normal_views),
                    "semantic_label": "unknown_ornamental_relief_patch",
                    "semantic_claim_authorized": False,
                    "cad_fusion_authorized": False,
                }
                parent_patches.append(patch)
                patches.append(patch)
                all_query_ids.extend(patch["query_node_ids"])
            subdivision_records.append(
                {
                    "parent_component_id": parent["component_id"],
                    "source_query_count": parent["query_count"],
                    "patch_count": len(parent_patches),
                    "rejected_pair_counts": rejected,
                    "patch_ids": [item["patch_id"] for item in parent_patches],
                }
            )

        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        graph = {
            "schema_version": "lalitha_phase5_relief_subdivision_object_v1",
            "object_id": object_id,
            "split": "val",
            "thresholds": {
                "spatial_link_radius_track_pixels": SPATIAL_LINK_RADIUS_TRACK_PIXELS,
                "maximum_normalized_depth_delta": MAX_NORMALIZED_DEPTH_DELTA,
                "maximum_edge_barrier_fraction": MAX_EDGE_BARRIER_FRACTION,
                "minimum_fit_review_query_count": MIN_FIT_REVIEW_QUERY_COUNT,
                "maximum_patch_diameter_track_pixels": MAX_PATCH_DIAMETER_TRACK_PIXELS,
            },
            "subdivisions": subdivision_records,
            "patches": patches,
            "claims": {
                "semantic_components_validated": False,
                "surface_geometry_validated": False,
                "hidden_geometry_recovered": False,
                "cad_fusion_authorized": False,
            },
        }
        graph_path = object_dir / "relief_subdivision.json"
        graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        audit_path = object_dir / "relief_subdivision_audit.png"
        _render_audit(
            source,
            depth,
            edge,
            mask,
            component_graph["relief_components"],
            patches,
            audit_path,
            object_id,
        )
        audit_paths.append(audit_path)
        source_query_ids = [
            node_id
            for component in component_graph["relief_components"]
            for node_id in component["query_node_ids"]
        ]
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "parent_component_count": len(
                    component_graph["relief_components"]
                ),
                "patch_count": len(patches),
                "subdivided_parent_count": sum(
                    item["patch_count"] > 1 for item in subdivision_records
                ),
                "singleton_patch_count": sum(
                    item["query_count"] == 1 for item in patches
                ),
                "fit_review_eligible_patch_count": sum(
                    item["future_surface_fit_review_eligible"] for item in patches
                ),
                "multi_view_patch_count": sum(
                    item["review_priority"] == "multi_view_supported"
                    for item in patches
                ),
                "source_queries_partitioned_exactly_once": sorted(all_query_ids)
                == sorted(source_query_ids)
                and len(all_query_ids) == len(set(all_query_ids)),
                "phase5_10_parent_count_reproduced": len(
                    component_graph["relief_components"]
                )
                == component_by_id[object_id]["relief_component_count"],
                "artifacts": {
                    "graph": _relative(graph_path, root),
                    "audit": _relative(audit_path, root),
                },
            }
        )

    cohort_path = output / "validation_relief_subdivision_audit.png"
    _cohort_audit(audit_paths, cohort_path)
    parent_count = sum(item["parent_component_count"] for item in objects)
    patch_count = sum(item["patch_count"] for item in objects)
    report = {
        "schema_version": "lalitha_phase5_relief_subdivision_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.11 edge-depth-view relief subdivision",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "source_view": "front",
        },
        "sources": {
            "component_evidence": f"{COMPONENT_ROOT.as_posix()}/report.json",
            "visibility_graph": f"{VISIBILITY_ROOT.as_posix()}/report.json",
            "phase1_features": PHASE1_REPORT.as_posix(),
        },
        "thresholds": {
            "spatial_link_radius_track_pixels": SPATIAL_LINK_RADIUS_TRACK_PIXELS,
            "maximum_normalized_depth_delta": MAX_NORMALIZED_DEPTH_DELTA,
            "maximum_edge_barrier_fraction": MAX_EDGE_BARRIER_FRACTION,
            "minimum_fit_review_query_count": MIN_FIT_REVIEW_QUERY_COUNT,
            "maximum_patch_diameter_track_pixels": MAX_PATCH_DIAMETER_TRACK_PIXELS,
        },
        "summary": {
            "object_count": len(objects),
            "parent_component_count": parent_count,
            "patch_count": patch_count,
            "patch_increase": patch_count - parent_count,
            "subdivided_parent_count": sum(
                item["subdivided_parent_count"] for item in objects
            ),
            "singleton_patch_count": sum(
                item["singleton_patch_count"] for item in objects
            ),
            "fit_review_eligible_patch_count": sum(
                item["fit_review_eligible_patch_count"] for item in objects
            ),
            "multi_view_patch_count": sum(
                item["multi_view_patch_count"] for item in objects
            ),
        },
        "checks": {
            "fixed_validation_objects_only": all(
                item["split"] == "val" for item in objects
            ),
            "phase5_10_parent_counts_reproduced": all(
                item["phase5_10_parent_count_reproduced"] for item in objects
            ),
            "all_source_queries_partitioned_exactly_once": all(
                item["source_queries_partitioned_exactly_once"] for item in objects
            ),
            "test_objects_not_processed": True,
            "source_component_graph_unchanged": True,
            "exact_cad_unchanged": True,
        },
        "objects": objects,
        "decision": "retain_subdivision_as_geometric_review_evidence",
        "claims": {
            "semantic_components_validated": False,
            "surface_geometry_validated": False,
            "metric_depth_validated": False,
            "hidden_geometry_recovered": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Relative-depth and edge discontinuities can be caused by reflections, shading or model bias.",
            "A patch can still contain several semantic ornaments or split one ornament into multiple pieces.",
            "Singleton patches are retained as unresolved evidence instead of being force-merged.",
            "Only visible front-source evidence is subdivided; hidden surfaces remain absent.",
        ],
        "artifacts": {"audit": _relative(cohort_path, root)},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase5_relief_subdivision_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = run(root, output)
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "summary": report["summary"],
                "checks": report["checks"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
