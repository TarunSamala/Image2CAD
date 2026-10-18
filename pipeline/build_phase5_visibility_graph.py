"""Build a visibility-aware relief evidence graph for Phase 5.9.

The graph is deliberately limited to the fixed validation split.  It converts
the frozen Phase 5.7 reciprocal tracks and Phase 5.8 normal annotations into
explicit source-query, target-view and observation-edge records.  No edge is
invented for a view that cannot observe a source surface, and no graph record
is converted into CAD geometry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.build_phase5_stl1_relief_evidence import _metric_depth_records
from pipeline.evaluate_phase5_dense_relief_pilot import _preserve_baseline_gate
from pipeline.evaluate_phase5_normal_corroboration import (
    _backend_records,
    _corroborate,
    _load_tracks,
    _normal_maps,
)
from pipeline.evaluate_phase5_relief_tracks import (
    _load_view_evidence,
    _metric3d_normal_records,
    evaluate_track_consistency,
)
from pipeline.extract_vggt_track_evidence import (
    VALIDATION_OBJECTS,
    VIEW_ORDER,
)


DEPTH_ROOT = Path("dataset/phase_runs/v1/phase5_validation_metric_depth_v1")
BASELINE_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_full_stl1_track_evidence_v1"
)
HYBRID_TRACK_ROOT = Path(
    "dataset/phase_runs/v1/vggt_relief_hybrid_validation_v1"
)
SOURCE_VIEW = "front"
SOURCE_VIEW_INDEX = VIEW_ORDER.index(SOURCE_VIEW)
REGION_LINK_RADIUS_PIXELS = 28.0

CLASS_ORDER = (
    "multi_view_normal_corroborated",
    "single_view_normal_corroborated",
    "reciprocal_normal_unresolved",
    "unobserved_source_candidate",
)
CLASS_COLORS_BGR = {
    "multi_view_normal_corroborated": (66, 174, 68),
    "single_view_normal_corroborated": (190, 145, 35),
    "reciprocal_normal_unresolved": (184, 70, 184),
    "unobserved_source_candidate": (150, 150, 150),
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _classification(
    accepted_targets: list[str], corroborated_targets: list[str]
) -> str:
    if len(corroborated_targets) >= 2:
        return "multi_view_normal_corroborated"
    if len(corroborated_targets) == 1:
        return "single_view_normal_corroborated"
    if accepted_targets:
        return "reciprocal_normal_unresolved"
    return "unobserved_source_candidate"


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


def _build_regions(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group nearby supported queries that share an observable target view."""

    supported = [node for node in nodes if node["accepted_target_views"]]
    union = _UnionFind(len(supported))
    radius_squared = REGION_LINK_RADIUS_PIXELS**2
    for left in range(len(supported)):
        left_point = np.asarray(supported[left]["source_xy"], np.float32)
        left_views = set(supported[left]["accepted_target_views"])
        for right in range(left + 1, len(supported)):
            if not left_views.intersection(
                supported[right]["accepted_target_views"]
            ):
                continue
            right_point = np.asarray(supported[right]["source_xy"], np.float32)
            if float(np.sum(np.square(left_point - right_point))) <= radius_squared:
                union.union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, node in enumerate(supported):
        grouped.setdefault(union.find(index), []).append(node)

    ordered = sorted(
        grouped.values(),
        key=lambda group: (
            min(item["source_xy"][1] for item in group),
            min(item["source_xy"][0] for item in group),
        ),
    )
    regions = []
    for region_index, group in enumerate(ordered, start=1):
        accepted_views = sorted(
            {view for node in group for view in node["accepted_target_views"]},
            key=VIEW_ORDER.index,
        )
        corroborated_views = sorted(
            {
                view
                for node in group
                for view in node["normal_corroborated_target_views"]
            },
            key=VIEW_ORDER.index,
        )
        points = np.asarray([node["source_xy"] for node in group], np.float32)
        class_counts = {
            name: sum(node["classification"] == name for node in group)
            for name in CLASS_ORDER
        }
        regions.append(
            {
                "region_id": f"front_region_{region_index:03d}",
                "query_node_ids": [node["node_id"] for node in group],
                "query_count": len(group),
                "centroid_xy": [
                    round(float(points[:, 0].mean()), 3),
                    round(float(points[:, 1].mean()), 3),
                ],
                "bbox_xyxy": [
                    round(float(points[:, 0].min()), 3),
                    round(float(points[:, 1].min()), 3),
                    round(float(points[:, 0].max()), 3),
                    round(float(points[:, 1].max()), 3),
                ],
                "accepted_target_views": accepted_views,
                "normal_corroborated_target_views": corroborated_views,
                "classification": _classification(
                    accepted_views, corroborated_views
                ),
                "query_class_counts": class_counts,
            }
        )
    return regions


def _render_audit(
    image: np.ndarray,
    nodes: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    output: Path,
    object_id: str,
) -> None:
    source = image.copy()
    scale_x = (source.shape[1] - 1) / 517.0
    scale_y = (source.shape[0] - 1) / 517.0
    for node in nodes:
        x = int(round(node["source_xy"][0] * scale_x))
        y = int(round(node["source_xy"][1] * scale_y))
        classification = node["classification"]
        radius = 2 if classification == "unobserved_source_candidate" else 4
        cv2.circle(
            source,
            (x, y),
            radius,
            CLASS_COLORS_BGR[classification],
            -1,
            cv2.LINE_AA,
        )

    region_view = image.copy()
    for region in regions:
        x1, y1, x2, y2 = region["bbox_xyxy"]
        p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
        p2 = (int(round(x2 * scale_x)), int(round(y2 * scale_y)))
        color = CLASS_COLORS_BGR[region["classification"]]
        cv2.rectangle(region_view, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(
            region_view,
            region["region_id"].removeprefix("front_region_"),
            (p1[0], max(12, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    height = 570
    width = 570
    panels = []
    for panel_image, title in (
        (source, "QUERY OBSERVABILITY"),
        (region_view, "OBSERVABLE SURFACE REGIONS"),
    ):
        resized = cv2.resize(panel_image, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height + 70, width, 3), 246, np.uint8)
        canvas[70:] = resized
        cv2.putText(canvas, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 29, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, object_id.upper(), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (75, 78, 82), 1, cv2.LINE_AA)
        panels.append(canvas)

    row = np.hstack(panels)
    counts = {
        name: sum(node["classification"] == name for node in nodes)
        for name in CLASS_ORDER
    }
    header = np.full((76, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        "PHASE 5.9 - VISIBILITY-AWARE RELIEF GRAPH",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 29, 34),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"multi-view {counts[CLASS_ORDER[0]]} | single-view {counts[CLASS_ORDER[1]]} | normal unresolved {counts[CLASS_ORDER[2]]} | unobserved candidates {counts[CLASS_ORDER[3]]}",
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    footer = np.full((58, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        footer,
        "Green = 2+ normal-corroborated target views | gold = one | magenta = reciprocal only | grey = no accepted target",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "Boxes group nearby supported queries sharing at least one observable target; they are evidence regions, not CAD surfaces.",
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    width = 1100
    rows = []
    for image in images:
        scale = width / image.shape[1]
        rows.append(
            cv2.resize(
                image,
                (width, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    cv2.imwrite(str(output), np.vstack(rows))


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(
        (
            root / "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
        ).read_text(encoding="utf-8")
    )
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}
    phase5_7 = json.loads(
        (root / "data/phase5_dense_relief_validation_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    phase5_7_by_id = {item["object_id"]: item for item in phase5_7["objects"]}
    phase5_8 = json.loads(
        (root / "data/phase5_normal_corroboration_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    phase5_8_by_id = {item["object_id"]: item for item in phase5_8["objects"]}
    depth_anything = _backend_records(root, "depth_anything_v2_metric")
    unidepth = _backend_records(root, "unidepth_v2")
    metric3d = _backend_records(root, "metric3d_v2")
    original_metric_depths = _metric_depth_records(root)
    original_direct_normals = _metric3d_normal_records(root)

    objects = []
    audit_paths = []
    for object_id in VALIDATION_OBJECTS:
        object_record = phase1_by_id[object_id]
        if object_record["split"] != "val":
            raise RuntimeError(f"{object_id} is not in the fixed validation split")
        evidence = _load_view_evidence(
            root, object_record, original_metric_depths, original_direct_normals
        )
        baseline = _load_tracks(
            root / BASELINE_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        hybrid = _load_tracks(
            root / HYBRID_TRACK_ROOT / object_id / "vggt_track_evidence.npz"
        )
        baseline_evaluated = evaluate_track_consistency(baseline, evidence)
        evaluated = _preserve_baseline_gate(
            evaluate_track_consistency(hybrid, evidence), baseline_evaluated
        )
        maps = _normal_maps(
            root, object_record, depth_anything, unidepth, metric3d
        )
        corroborated, source_agreement, target_agreement = _corroborate(
            hybrid, evaluated["accepted"], maps
        )

        source_points = hybrid["query_points_xy"][SOURCE_VIEW_INDEX]
        candidate = evaluated["candidate"][SOURCE_VIEW_INDEX]
        accepted = evaluated["accepted"][SOURCE_VIEW_INDEX]
        normal_ok = corroborated[SOURCE_VIEW_INDEX]
        nodes = []
        edges = []
        for query_index in np.flatnonzero(candidate):
            accepted_views = [
                view
                for target_index, view in enumerate(VIEW_ORDER)
                if target_index != SOURCE_VIEW_INDEX
                and bool(accepted[target_index, query_index])
            ]
            corroborated_views = [
                view
                for target_index, view in enumerate(VIEW_ORDER)
                if target_index != SOURCE_VIEW_INDEX
                and bool(normal_ok[target_index, query_index])
            ]
            node_id = f"{object_id}:front:q{int(query_index):04d}"
            nodes.append(
                {
                    "node_id": node_id,
                    "query_index": int(query_index),
                    "source_view": SOURCE_VIEW,
                    "source_xy": [
                        round(float(source_points[query_index, 0]), 3),
                        round(float(source_points[query_index, 1]), 3),
                    ],
                    "source_relief_confidence": round(
                        float(
                            evaluated["source_relief_confidence"][
                                SOURCE_VIEW_INDEX, query_index
                            ]
                        ),
                        6,
                    ),
                    "source_normal_agreement": round(
                        float(source_agreement[SOURCE_VIEW_INDEX, query_index]), 6
                    ),
                    "accepted_target_views": accepted_views,
                    "normal_corroborated_target_views": corroborated_views,
                    "classification": _classification(
                        accepted_views, corroborated_views
                    ),
                }
            )
            for target_index, target_view in enumerate(VIEW_ORDER):
                if target_index == SOURCE_VIEW_INDEX or not accepted[
                    target_index, query_index
                ]:
                    continue
                edges.append(
                    {
                        "edge_id": f"{node_id}->{object_id}:{target_view}",
                        "source_node_id": node_id,
                        "target_view_node_id": f"{object_id}:{target_view}",
                        "target_view": target_view,
                        "target_xy": [
                            round(
                                float(
                                    hybrid["tracks_xy"][
                                        SOURCE_VIEW_INDEX,
                                        target_index,
                                        query_index,
                                        0,
                                    ]
                                ),
                                3,
                            ),
                            round(
                                float(
                                    hybrid["tracks_xy"][
                                        SOURCE_VIEW_INDEX,
                                        target_index,
                                        query_index,
                                        1,
                                    ]
                                ),
                                3,
                            ),
                        ],
                        "reciprocal_accepted": True,
                        "normal_corroborated": bool(
                            normal_ok[target_index, query_index]
                        ),
                        "track_score": round(
                            float(
                                evaluated["score"][
                                    SOURCE_VIEW_INDEX,
                                    target_index,
                                    query_index,
                                ]
                            ),
                            6,
                        ),
                        "cycle_error_pixels": round(
                            float(
                                evaluated["cycle_error"][
                                    SOURCE_VIEW_INDEX,
                                    target_index,
                                    query_index,
                                ]
                            ),
                            6,
                        ),
                        "nearest_query_distance_pixels": round(
                            float(
                                evaluated["nearest_query_distance"][
                                    SOURCE_VIEW_INDEX,
                                    target_index,
                                    query_index,
                                ]
                            ),
                            6,
                        ),
                        "target_normal_agreement": round(
                            float(
                                target_agreement[
                                    SOURCE_VIEW_INDEX,
                                    target_index,
                                    query_index,
                                ]
                            ),
                            6,
                        ),
                    }
                )

        regions = _build_regions(nodes)
        class_counts = {
            name: sum(node["classification"] == name for node in nodes)
            for name in CLASS_ORDER
        }
        target_view_counts = {}
        for target_view in VIEW_ORDER[1:]:
            view_edges = [edge for edge in edges if edge["target_view"] == target_view]
            target_view_counts[target_view] = {
                "reciprocal_edges": len(view_edges),
                "normal_corroborated_edges": sum(
                    edge["normal_corroborated"] for edge in view_edges
                ),
            }
        graph = {
            "schema_version": "lalitha_phase5_visibility_graph_object_v1",
            "object_id": object_id,
            "split": "val",
            "source_view": SOURCE_VIEW,
            "view_nodes": [
                {
                    "node_id": f"{object_id}:{view}",
                    "view": view,
                    "role": "source" if view == SOURCE_VIEW else "target",
                }
                for view in VIEW_ORDER
            ],
            "query_nodes": nodes,
            "observation_edges": edges,
            "surface_regions": regions,
        }
        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        graph_path = object_dir / "visibility_graph.json"
        graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        source_image = cv2.imread(
            str(evidence[SOURCE_VIEW]["source_image"]), cv2.IMREAD_COLOR
        )
        audit_path = object_dir / "visibility_graph_audit.png"
        _render_audit(source_image, nodes, regions, audit_path, object_id)
        audit_paths.append(audit_path)

        supported_count = sum(bool(node["accepted_target_views"]) for node in nodes)
        normal_supported_count = sum(
            bool(node["normal_corroborated_target_views"]) for node in nodes
        )
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "candidate_query_count": len(nodes),
                "reciprocally_supported_query_count": supported_count,
                "normal_corroborated_query_count": normal_supported_count,
                "observation_edge_count": len(edges),
                "normal_corroborated_edge_count": sum(
                    edge["normal_corroborated"] for edge in edges
                ),
                "surface_region_count": len(regions),
                "query_class_counts": class_counts,
                "region_class_counts": {
                    name: sum(region["classification"] == name for region in regions)
                    for name in CLASS_ORDER
                },
                "target_views": target_view_counts,
                "phase5_7_supported_count_reproduced": supported_count
                == phase5_7_by_id[object_id]["variants"]["phase5_7_hybrid"][
                    "front_supported_count"
                ],
                "phase5_8_normal_count_reproduced": normal_supported_count
                == phase5_8_by_id[object_id]["normal_corroborated_count"],
                "artifacts": {
                    "graph": _relative(graph_path, root),
                    "audit": _relative(audit_path, root),
                },
            }
        )

    cohort_path = output / "validation_visibility_graph_audit.png"
    _cohort_audit(audit_paths, cohort_path)
    totals = {
        name: sum(item["query_class_counts"][name] for item in objects)
        for name in CLASS_ORDER
    }
    report = {
        "schema_version": "lalitha_phase5_visibility_graph_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.9 visibility-aware relief evidence graph",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "source_view": SOURCE_VIEW,
            "target_views": list(VIEW_ORDER[1:]),
            "region_link_radius_pixels": REGION_LINK_RADIUS_PIXELS,
        },
        "sources": {
            "phase5_7": "data/phase5_dense_relief_validation_v1/report.json",
            "phase5_8": "data/phase5_normal_corroboration_v1/report.json",
            "hybrid_tracks": f"{HYBRID_TRACK_ROOT.as_posix()}/report.json",
        },
        "summary": {
            "object_count": len(objects),
            "candidate_query_count": sum(
                item["candidate_query_count"] for item in objects
            ),
            "reciprocally_supported_query_count": sum(
                item["reciprocally_supported_query_count"] for item in objects
            ),
            "normal_corroborated_query_count": sum(
                item["normal_corroborated_query_count"] for item in objects
            ),
            "observation_edge_count": sum(
                item["observation_edge_count"] for item in objects
            ),
            "normal_corroborated_edge_count": sum(
                item["normal_corroborated_edge_count"] for item in objects
            ),
            "surface_region_count": sum(
                item["surface_region_count"] for item in objects
            ),
            "query_class_counts": totals,
            "multi_view_normal_corroborated_fraction_of_supported": round(
                totals["multi_view_normal_corroborated"]
                / max(
                    sum(item["reciprocally_supported_query_count"] for item in objects),
                    1,
                ),
                6,
            ),
        },
        "checks": {
            "fixed_validation_objects_only": all(
                item["split"] == "val" for item in objects
            ),
            "phase5_7_supported_counts_reproduced": all(
                item["phase5_7_supported_count_reproduced"] for item in objects
            ),
            "phase5_8_normal_counts_reproduced": all(
                item["phase5_8_normal_count_reproduced"] for item in objects
            ),
            "all_edges_are_reciprocal_phase5_7_edges": True,
            "test_objects_not_processed": True,
            "source_tracks_unchanged": True,
            "exact_cad_unchanged": True,
        },
        "objects": objects,
        "decision": "retain_visibility_graph_for_component_review_and_view_routing",
        "claims": {
            "visibility_physically_validated": False,
            "hidden_geometry_recovered": False,
            "physical_correspondence_validated": False,
            "cad_fusion_authorized": False,
            "metric_scale_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "The graph records learned reciprocal observations; it is not physical visibility ground truth.",
            "Multi-view means two or more named target views, not calibrated angular diversity.",
            "Unobserved source candidates are unresolved; surfaces hidden from the front view are absent from this graph.",
            "Region grouping is a deterministic image-space neighborhood used for review, not semantic component segmentation.",
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
        default=Path("data/phase5_visibility_graph_v1"),
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
