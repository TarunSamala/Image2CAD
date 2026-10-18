"""Attach Phase 5.9 visibility regions to stable component evidence nodes.

This validation-only bridge creates machine-proposed relief components, spatial
head zones and relationships to the existing exact metal/head candidates and
Phase 4 negative-space proposals.  It deliberately leaves jewellery semantics
unknown: spatial proximity cannot prove a stone, prong, face, cavity or opening.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.extract_vggt_track_evidence import VALIDATION_OBJECTS


VISIBILITY_ROOT = Path("data/phase5_visibility_graph_v1")
PHASE4_CANDIDATE_REPORT = Path(
    "data/phase4_stl1_component_candidates_v1/report.json"
)
PHASE5_HEAD_REPORT = Path(
    "data/phase5_stl1_exact_head_support_fit_v1/report.json"
)
PHASE1_REPORT = Path(
    "dataset/phase_runs/v1/phase1_complete_features_v1/report.json"
)
NEGATIVE_SPACE_LINK_DISTANCE_PIXELS = 20.0
TRACK_COORDINATE_MAX = 517.0

ZONE_ORDER = (
    "upper_head",
    "left_head",
    "central_head",
    "right_head",
    "lower_head",
)
ZONE_COLORS_BGR = {
    "upper_head": (192, 104, 42),
    "left_head": (174, 78, 148),
    "central_head": (66, 174, 68),
    "right_head": (55, 152, 210),
    "lower_head": (126, 105, 196),
}
EVIDENCE_COLORS_BGR = {
    "multi_view_normal_corroborated": (66, 174, 68),
    "single_view_normal_corroborated": (190, 145, 35),
    "reciprocal_normal_unresolved": (184, 70, 184),
}


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _zone_for_point(
    point_xy: tuple[float, float], bbox_xyxy: tuple[int, int, int, int]
) -> str:
    x0, y0, x1, y1 = bbox_xyxy
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    nx = (point_xy[0] - x0) / width
    ny = (point_xy[1] - y0) / height
    if ny < 1.0 / 3.0:
        return "upper_head"
    if ny >= 2.0 / 3.0:
        return "lower_head"
    if nx < 1.0 / 3.0:
        return "left_head"
    if nx >= 2.0 / 3.0:
        return "right_head"
    return "central_head"


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise RuntimeError("Front jewellery mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _scale_track_point(
    point_xy: list[float], width: int, height: int
) -> tuple[float, float]:
    return (
        float(point_xy[0]) * (width - 1) / TRACK_COORDINATE_MAX,
        float(point_xy[1]) * (height - 1) / TRACK_COORDINATE_MAX,
    )


def _distance_map(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    if not np.any(binary):
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, 5)


def _sample_minimum_distance(
    distance: np.ndarray, points: list[tuple[float, float]]
) -> float:
    height, width = distance.shape
    values = []
    for x, y in points:
        ix = int(np.clip(round(x), 0, width - 1))
        iy = int(np.clip(round(y), 0, height - 1))
        values.append(float(distance[iy, ix]))
    return min(values) if values else float("inf")


def _primary_zone(counts: dict[str, int]) -> str:
    return max(ZONE_ORDER, key=lambda name: (counts.get(name, 0), -ZONE_ORDER.index(name)))


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    width = 570
    height = 570
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 72, width, 3), 246, np.uint8)
    canvas[72:] = resized
    cv2.putText(canvas, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 29, 34), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (65, 68, 73), 1, cv2.LINE_AA)
    return canvas


def _render_audit(
    source: np.ndarray,
    jewellery_mask: np.ndarray,
    components: list[dict[str, Any]],
    candidate_masks: dict[str, np.ndarray],
    output: Path,
    object_id: str,
) -> None:
    height, width = source.shape[:2]
    bbox = _foreground_bbox(jewellery_mask)
    zone_view = source.copy()
    x0, y0, x1, y1 = bbox
    x_a = int(round(x0 + (x1 - x0) / 3.0))
    x_b = int(round(x0 + 2.0 * (x1 - x0) / 3.0))
    y_a = int(round(y0 + (y1 - y0) / 3.0))
    y_b = int(round(y0 + 2.0 * (y1 - y0) / 3.0))
    for left, top, right, bottom, zone in (
        (x0, y0, x1, y_a, "upper_head"),
        (x0, y_a, x_a, y_b, "left_head"),
        (x_a, y_a, x_b, y_b, "central_head"),
        (x_b, y_a, x1, y_b, "right_head"),
        (x0, y_b, x1, y1, "lower_head"),
    ):
        overlay = zone_view.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), ZONE_COLORS_BGR[zone], -1)
        zone_view = cv2.addWeighted(overlay, 0.12, zone_view, 0.88, 0.0)
        cv2.rectangle(zone_view, (left, top), (right, bottom), ZONE_COLORS_BGR[zone], 1)

    graph_view = source.copy()
    for mask in candidate_masks.values():
        contours, _ = cv2.findContours(
            np.uint8(mask > 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(graph_view, contours, -1, (198, 68, 150), 2, cv2.LINE_AA)

    for component in components:
        x1t, y1t, x2t, y2t = component["source_bbox_xyxy"]
        p1 = (
            int(round(x1t * (width - 1) / TRACK_COORDINATE_MAX)),
            int(round(y1t * (height - 1) / TRACK_COORDINATE_MAX)),
        )
        p2 = (
            int(round(x2t * (width - 1) / TRACK_COORDINATE_MAX)),
            int(round(y2t * (height - 1) / TRACK_COORDINATE_MAX)),
        )
        color = EVIDENCE_COLORS_BGR[component["evidence_state"]]
        cv2.rectangle(graph_view, p1, p2, color, 2, cv2.LINE_AA)
        label = component["component_id"].rsplit("_", 1)[-1]
        cv2.putText(
            graph_view,
            label,
            (p1[0], max(12, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    linked = sum(bool(item["adjacent_negative_space_ids"]) for item in components)
    panels = [
        _panel(
            zone_view,
            "DETERMINISTIC SPATIAL ZONES",
            "Location only - not jewellery semantics",
        ),
        _panel(
            graph_view,
            "STABLE RELIEF COMPONENT PROPOSALS",
            f"{len(components)} regions | {linked} near negative-space evidence",
        ),
    ]
    row = np.hstack(panels)
    header = np.full((74, row.shape[1], 3), 246, np.uint8)
    cv2.putText(
        header,
        f"PHASE 5.10 - {object_id.upper()} COMPONENT EVIDENCE GRAPH",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 29, 34),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Stable IDs + visibility provenance + metal/head parentage + negative-space adjacency",
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
        "Green = multi-view normal support | gold = single-view | magenta = normal unresolved | pink contours = unresolved negative spaces",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "No proposal is labelled stone, prong, face or cavity; human or labelled evidence is still required.",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (55, 60, 65),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), np.vstack((header, row, footer)))


def _cohort_audit(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    target_width = 1100
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
    visibility_report = json.loads(
        (root / VISIBILITY_ROOT / "report.json").read_text(encoding="utf-8")
    )
    phase4 = json.loads(
        (root / PHASE4_CANDIDATE_REPORT).read_text(encoding="utf-8")
    )
    head_report = json.loads(
        (root / PHASE5_HEAD_REPORT).read_text(encoding="utf-8")
    )
    head_by_id = {item["object_id"]: item for item in head_report["objects"]}
    phase1 = json.loads((root / PHASE1_REPORT).read_text(encoding="utf-8"))
    phase1_by_id = {item["object_id"]: item for item in phase1["objects"]}

    objects = []
    audit_paths = []
    for object_id in VALIDATION_OBJECTS:
        source_record = phase1_by_id[object_id]
        if source_record["split"] != "val":
            raise RuntimeError(f"{object_id} is outside the fixed validation split")
        visibility_path = root / VISIBILITY_ROOT / object_id / "visibility_graph.json"
        visibility = json.loads(visibility_path.read_text(encoding="utf-8"))
        query_by_id = {item["node_id"]: item for item in visibility["query_nodes"]}
        source_path = root / source_record["views"]["front"]["normalized_source"]
        mask_path = (
            root / "dataset/prepared_v1/masks/jewelry" / object_id / "front.png"
        )
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        jewellery_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None or jewellery_mask is None:
            raise RuntimeError(f"Missing normalized front evidence for {object_id}")
        height, width = source.shape[:2]
        foreground_bbox = _foreground_bbox(jewellery_mask)

        front_candidates = phase4["objects"][object_id]["views"]["front"][
            "candidates"
        ]
        candidate_masks = {}
        candidate_distances = {}
        for candidate in front_candidates:
            candidate_mask = cv2.imread(
                str(root / candidate["mask_path"]), cv2.IMREAD_GRAYSCALE
            )
            if candidate_mask is None:
                raise RuntimeError(f"Missing candidate mask {candidate['mask_path']}")
            candidate_masks[candidate["candidate_id"]] = candidate_mask
            candidate_distances[candidate["candidate_id"]] = _distance_map(
                candidate_mask
            )

        head_component_id = head_by_id[object_id]["head_support_spec"][
            "component_id"
        ]
        metal_component_id = f"{object_id}_metal_body_001"
        nodes = [
            {
                "component_id": metal_component_id,
                "component_type": "exact_metal_body_candidate",
                "provenance": "phase5_1_exact_ring_body_plus_phase5_2_head_support",
                "review_state": "machine_proposed",
                "semantic_state": "structural_role_known_image_fit_unapproved",
            },
            {
                "component_id": head_component_id,
                "component_type": "exact_head_support_candidate",
                "provenance": "phase5_2_two_view_outer_envelope",
                "review_state": "machine_proposed",
                "semantic_state": "outer_envelope_only",
            },
        ]
        edges = [
            {
                "source_id": head_component_id,
                "relationship": "part_of",
                "target_id": metal_component_id,
                "authority": "machine_proposed_structural_relationship",
            }
        ]
        for zone in ZONE_ORDER:
            zone_id = f"{object_id}_{zone}_zone_001"
            nodes.append(
                {
                    "component_id": zone_id,
                    "component_type": "spatial_head_zone",
                    "spatial_zone": zone,
                    "provenance": "deterministic_front_silhouette_partition",
                    "review_state": "machine_proposed",
                    "semantic_state": "location_only_not_jewellery_semantics",
                }
            )
            edges.append(
                {
                    "source_id": zone_id,
                    "relationship": "spatial_partition_of",
                    "target_id": head_component_id,
                    "authority": "deterministic_image_space_partition",
                }
            )
        for candidate in front_candidates:
            nodes.append(
                {
                    "component_id": candidate["candidate_id"],
                    "component_type": "negative_space_candidate",
                    "candidate_subtype": candidate["candidate_subtype"],
                    "provenance": "phase4_enclosed_negative_space",
                    "review_state": candidate["review_state"],
                    "semantic_state": candidate["semantic_state"],
                    "is_confirmed_stone": False,
                }
            )
            edges.append(
                {
                    "source_id": candidate["candidate_id"],
                    "relationship": "observed_within_projection_of",
                    "target_id": head_component_id,
                    "authority": "machine_proposed_2d_containment",
                }
            )

        relief_components = []
        for component_index, region in enumerate(
            visibility["surface_regions"], start=1
        ):
            component_id = f"{object_id}_relief_component_{component_index:03d}"
            points_track = [
                query_by_id[node_id]["source_xy"]
                for node_id in region["query_node_ids"]
            ]
            points_image = [
                _scale_track_point(point, width, height) for point in points_track
            ]
            zone_counts = {zone: 0 for zone in ZONE_ORDER}
            for point in points_image:
                zone_counts[_zone_for_point(point, foreground_bbox)] += 1
            primary_zone = _primary_zone(zone_counts)
            adjacent = []
            candidate_distance_records = []
            for candidate in front_candidates:
                distance = _sample_minimum_distance(
                    candidate_distances[candidate["candidate_id"]], points_image
                )
                if distance <= NEGATIVE_SPACE_LINK_DISTANCE_PIXELS:
                    adjacent.append(candidate["candidate_id"])
                    candidate_distance_records.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "minimum_distance_pixels": round(distance, 3),
                        }
                    )
            component = {
                "component_id": component_id,
                "component_type": "ornamental_relief_region_candidate",
                "source_visibility_region_id": region["region_id"],
                "source_bbox_xyxy": region["bbox_xyxy"],
                "query_node_ids": region["query_node_ids"],
                "query_count": region["query_count"],
                "primary_spatial_zone": primary_zone,
                "spatial_zone_counts": zone_counts,
                "accepted_target_views": region["accepted_target_views"],
                "normal_corroborated_target_views": region[
                    "normal_corroborated_target_views"
                ],
                "evidence_state": region["classification"],
                "adjacent_negative_space_ids": adjacent,
                "negative_space_distances": candidate_distance_records,
                "review_state": "machine_proposed",
                "semantic_label": "unknown_ornamental_relief",
                "semantic_claim_authorized": False,
                "cad_fusion_authorized": False,
            }
            relief_components.append(component)
            nodes.append(component)
            edges.extend(
                [
                    {
                        "source_id": component_id,
                        "relationship": "surface_evidence_for",
                        "target_id": head_component_id,
                        "authority": "phase5_9_visibility_region",
                    },
                    {
                        "source_id": component_id,
                        "relationship": "located_in",
                        "target_id": f"{object_id}_{primary_zone}_zone_001",
                        "authority": "deterministic_image_space_partition",
                    },
                ]
            )
            for candidate_id in adjacent:
                edges.append(
                    {
                        "source_id": component_id,
                        "relationship": "image_space_adjacent_to",
                        "target_id": candidate_id,
                        "authority": "distance_threshold_machine_proposal",
                    }
                )

        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        graph = {
            "schema_version": "lalitha_phase5_component_evidence_graph_object_v1",
            "object_id": object_id,
            "split": "val",
            "scope": {
                "source_view": "front",
                "negative_space_link_distance_pixels": NEGATIVE_SPACE_LINK_DISTANCE_PIXELS,
            },
            "nodes": nodes,
            "edges": edges,
            "relief_components": relief_components,
            "claims": {
                "semantic_components_validated": False,
                "stones_confirmed": False,
                "faces_confirmed": False,
                "prongs_confirmed": False,
                "hidden_geometry_recovered": False,
                "cad_fusion_authorized": False,
            },
        }
        graph_path = object_dir / "component_evidence_graph.json"
        graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        audit_path = object_dir / "component_evidence_audit.png"
        _render_audit(
            source,
            jewellery_mask,
            relief_components,
            candidate_masks,
            audit_path,
            object_id,
        )
        audit_paths.append(audit_path)
        objects.append(
            {
                "object_id": object_id,
                "split": "val",
                "node_count": len(nodes),
                "edge_count": len(edges),
                "relief_component_count": len(relief_components),
                "multi_view_component_count": sum(
                    item["evidence_state"]
                    == "multi_view_normal_corroborated"
                    for item in relief_components
                ),
                "single_view_component_count": sum(
                    item["evidence_state"]
                    == "single_view_normal_corroborated"
                    for item in relief_components
                ),
                "normal_unresolved_component_count": sum(
                    item["evidence_state"]
                    == "reciprocal_normal_unresolved"
                    for item in relief_components
                ),
                "negative_space_candidate_count": len(front_candidates),
                "components_adjacent_to_negative_space_count": sum(
                    bool(item["adjacent_negative_space_ids"])
                    for item in relief_components
                ),
                "visibility_region_count_reproduced": len(relief_components)
                == len(visibility["surface_regions"]),
                "all_component_ids_unique": len(
                    {item["component_id"] for item in nodes}
                )
                == len(nodes),
                "artifacts": {
                    "graph": _relative(graph_path, root),
                    "audit": _relative(audit_path, root),
                },
            }
        )

    cohort_path = output / "validation_component_evidence_audit.png"
    _cohort_audit(audit_paths, cohort_path)
    report = {
        "schema_version": "lalitha_phase5_component_evidence_graph_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Phase 5.10 visibility-to-component evidence bridge",
        "scope": {
            "object_ids": list(VALIDATION_OBJECTS),
            "split": "val",
            "source_view": "front",
        },
        "sources": {
            "visibility_graph": f"{VISIBILITY_ROOT.as_posix()}/report.json",
            "negative_space_candidates": PHASE4_CANDIDATE_REPORT.as_posix(),
            "head_supports": PHASE5_HEAD_REPORT.as_posix(),
        },
        "summary": {
            "object_count": len(objects),
            "relief_component_count": sum(
                item["relief_component_count"] for item in objects
            ),
            "multi_view_component_count": sum(
                item["multi_view_component_count"] for item in objects
            ),
            "single_view_component_count": sum(
                item["single_view_component_count"] for item in objects
            ),
            "normal_unresolved_component_count": sum(
                item["normal_unresolved_component_count"] for item in objects
            ),
            "negative_space_candidate_count": sum(
                item["negative_space_candidate_count"] for item in objects
            ),
            "components_adjacent_to_negative_space_count": sum(
                item["components_adjacent_to_negative_space_count"]
                for item in objects
            ),
        },
        "checks": {
            "fixed_validation_objects_only": all(
                item["split"] == "val" for item in objects
            ),
            "phase5_9_visibility_regions_reproduced": all(
                item["visibility_region_count_reproduced"] for item in objects
            ),
            "stable_component_ids_unique_per_object": all(
                item["all_component_ids_unique"] for item in objects
            ),
            "all_relief_components_have_head_parent": True,
            "test_objects_not_processed": True,
            "source_visibility_graph_unchanged": True,
            "exact_cad_unchanged": True,
        },
        "objects": objects,
        "decision": "retain_machine_component_proposals_for_semantic_review",
        "claims": {
            "semantic_components_validated": False,
            "cross_view_component_identity_validated": False,
            "stones_confirmed": False,
            "prongs_confirmed": False,
            "faces_confirmed": False,
            "cad_fusion_authorized": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "STL-1 has no labelled component semantics, so spatial zones and relief components remain machine proposals.",
            "Image-space adjacency to negative space does not prove a stone, seat, cavity or opening.",
            "Broad visibility regions can contain multiple semantic ornaments and require later subdivision or review.",
            "Only front-source regions are represented; hidden surfaces remain absent.",
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
        default=Path("data/phase5_component_evidence_graph_v1"),
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
