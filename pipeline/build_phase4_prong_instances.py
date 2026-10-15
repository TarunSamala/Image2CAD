"""Build review-gated Ring01 prong instances from retained image masks.

The stage separates four machine-proposed instances only where the image-plane
evidence supports it.  Side/back projections remain explicit overlap groups;
they are never duplicated and presented as four independently observed prongs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("front", "side", "top", "angled", "back")
RESOLVED_VIEWS = {"front", "top", "angled"}
COMPONENT_IDS = tuple(f"prong_{index:03d}" for index in range(1, 5))
COLORS_BGR = (
    (56, 56, 230),
    (42, 174, 255),
    (210, 120, 40),
    (180, 70, 190),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_image(path: Path, mode: int) -> np.ndarray:
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _binary(path: Path) -> np.ndarray:
    return (_read_image(path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) * 255


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask, binaryImage=True)
    if not moments["m00"]:
        raise ValueError("Cannot calculate a centroid for an empty mask")
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _mask_metrics(mask: np.ndarray) -> dict[str, Any]:
    points = cv2.findNonZero(mask)
    if points is None:
        return {"area_px2": 0, "bbox_px": None, "center_px": None}
    x, y, width, height = cv2.boundingRect(points)
    center_x, center_y = _centroid(mask)
    return {
        "area_px2": int(np.count_nonzero(mask)),
        "bbox_px": {"x": x, "y": y, "width": width, "height": height},
        "center_px": {"x": round(center_x, 3), "y": round(center_y, 3)},
    }


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.count_nonzero((left > 0) & (right > 0))
    union = np.count_nonzero((left > 0) | (right > 0))
    return float(intersection / union) if union else 1.0


def _principal_components(mask: np.ndarray, expected: int = 4) -> list[np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    candidates = sorted(
        range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True
    )
    if len(candidates) < expected:
        raise ValueError(f"Expected {expected} prong regions, found {len(candidates)}")
    principal = candidates[:expected]
    centers = np.asarray([centroids[index] for index in principal], dtype=np.float32)
    ys, xs = np.nonzero(mask > 0)
    pixels = np.column_stack((xs, ys)).astype(np.float32)
    nearest = np.argmin(((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    results: list[np.ndarray] = []
    for region_index in range(expected):
        result = np.zeros_like(mask)
        selected = nearest == region_index
        result[ys[selected], xs[selected]] = 255
        results.append(result)
    return results


def _assign_resolved_ids(
    view: str, masks: list[np.ndarray], stone_center: tuple[float, float]
) -> dict[str, np.ndarray]:
    records = [(mask, *_centroid(mask)) for mask in masks]
    if view in {"front", "top"}:
        top = sorted(records, key=lambda item: item[2])[:2]
        bottom = sorted(records, key=lambda item: item[2])[2:]
        top.sort(key=lambda item: item[1])
        bottom.sort(key=lambda item: item[1])
        ordered = [top[0], top[1], bottom[0], bottom[1]]
        return {component_id: record[0] for component_id, record in zip(COMPONENT_IDS, ordered)}

    center_x, center_y = stone_center
    clockwise = sorted(
        records,
        key=lambda item: (math.atan2(item[1] - center_x, center_y - item[2]) + 2 * math.pi)
        % (2 * math.pi),
    )
    upper_index = min(range(len(clockwise)), key=lambda index: clockwise[index][2])
    clockwise = clockwise[upper_index:] + clockwise[:upper_index]
    # The angled reference rotates the four image-plane corners into a diamond.
    # Clockwise from the uppermost proposal maps TL, TR, BR, BL.
    clockwise_ids = ("prong_001", "prong_002", "prong_004", "prong_003")
    return {component_id: record[0] for component_id, record in zip(clockwise_ids, clockwise)}


def _split_overlap_groups(
    mask: np.ndarray, stone_center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    center_x, _ = stone_center
    columns = np.arange(mask.shape[1])[None, :]
    left = np.where((mask > 0) & (columns <= center_x), 255, 0).astype(np.uint8)
    right = np.where((mask > 0) & (columns > center_x), 255, 0).astype(np.uint8)
    if not np.any(left) or not np.any(right):
        raise ValueError("Projected prong overlap could not be split into two evidence groups")
    return left, right


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"Could not write {path}")
    return {"path": str(path), "sha256": _sha256(path), **_mask_metrics(mask)}


def _relative(path: str | Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root))


def _make_audit(
    source: np.ndarray,
    aggregate: np.ndarray,
    layers: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str,
) -> np.ndarray:
    overlay = source.copy()
    for _, mask, color in layers:
        active = mask > 0
        overlay[active] = (overlay[active].astype(np.float32) * 0.35 + np.asarray(color) * 0.65).astype(np.uint8)
    aggregate_panel = cv2.cvtColor(aggregate, cv2.COLOR_GRAY2BGR)
    panels = [source, aggregate_panel, overlay]
    labels = [f"{title.upper()} SOURCE", "AGGREGATE PRONG MASK", "INSTANCE / OVERLAP PROPOSAL"]
    labelled = []
    for panel, label in zip(panels, labels):
        canvas = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
        cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        labelled.append(canvas)
    return cv2.hconcat(labelled)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    source_dir = root / "data/ring01_reference_images"
    phase2_dir = root / "data/ring01_phase2_refined/masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    view_reports: dict[str, Any] = {}
    audit_rows: list[np.ndarray] = []

    for view in VIEWS:
        source = _read_image(source_dir / f"ring01_{view}.png", cv2.IMREAD_COLOR)
        aggregate = _binary(phase2_dir / "prongs" / f"ring01_{view}_prongs.png")
        stone = _binary(phase2_dir / "stone_amodal" / f"ring01_{view}_stone_amodal.png")
        stone_center = _centroid(stone)
        component_observations: dict[str, Any] = {}
        layers: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []

        if view in RESOLVED_VIEWS:
            assigned = _assign_resolved_ids(view, _principal_components(aggregate), stone_center)
            union = np.zeros_like(aggregate)
            for color_index, component_id in enumerate(COMPONENT_IDS):
                mask = assigned[component_id]
                path = output_dir / "masks" / view / f"ring01_{view}_{component_id}.png"
                artifact = _write_mask(path, mask)
                artifact["path"] = _relative(artifact["path"], root)
                component_observations[component_id] = {
                    "visible_mask": artifact["path"],
                    "amodal_mask": None,
                    "sha256": artifact["sha256"],
                    "confidence": 0.78 if view in {"front", "top"} else 0.58,
                    "provenance_state": "inferred",
                    "review_state": "pending",
                    "aggregate_instance_mask": False,
                    "visibility_state": "resolved_machine_proposal",
                    "identity_method": "image_plane_quadrants" if view in {"front", "top"} else "clockwise_order_upper_anchor",
                    **{key: artifact[key] for key in ("area_px2", "bbox_px", "center_px")},
                }
                union = cv2.bitwise_or(union, mask)
                layers.append((component_id, mask, COLORS_BGR[color_index]))
            resolved = True
            ambiguity = None
        else:
            left, right = _split_overlap_groups(aggregate, stone_center)
            groups = {
                "left_projection_group": (left, ("prong_001", "prong_003")),
                "right_projection_group": (right, ("prong_002", "prong_004")),
            }
            union = np.zeros_like(aggregate)
            for group_index, (group_id, (mask, candidates)) in enumerate(groups.items()):
                path = output_dir / "groups" / view / f"ring01_{view}_{group_id}.png"
                artifact = _write_mask(path, mask)
                artifact["path"] = _relative(artifact["path"], root)
                for component_id in candidates:
                    component_observations[component_id] = {
                        "visible_mask": artifact["path"],
                        "amodal_mask": None,
                        "sha256": artifact["sha256"],
                        "confidence": 0.35,
                        "provenance_state": "inferred",
                        "review_state": "pending",
                        "aggregate_instance_mask": True,
                        "visibility_state": "unresolved_projected_overlap",
                        "overlap_group_id": group_id,
                        "candidate_component_ids": list(candidates),
                        "identity_method": "left_right_projection_group_only",
                        **{key: artifact[key] for key in ("area_px2", "bbox_px", "center_px")},
                    }
                union = cv2.bitwise_or(union, mask)
                layers.append((group_id, mask, COLORS_BGR[group_index]))
            resolved = False
            ambiguity = "Two physical prong pairs overlap in this projection; individual identity is not observable from this image alone."

        pairwise_overlap = 0
        layer_masks = [item[1] for item in layers]
        for left_index, left_mask in enumerate(layer_masks):
            for right_mask in layer_masks[left_index + 1 :]:
                pairwise_overlap += int(np.count_nonzero((left_mask > 0) & (right_mask > 0)))
        view_reports[view] = {
            "identity_status": "resolved_machine_proposal" if resolved else "ambiguous_projection_groups",
            "resolved_individual_instances": resolved,
            "aggregate_area_px2": int(np.count_nonzero(aggregate)),
            "proposal_union_area_px2": int(np.count_nonzero(union)),
            "aggregate_union_iou": round(_iou(aggregate, union), 6),
            "pairwise_overlap_px2": pairwise_overlap,
            "component_observations": component_observations,
            "ambiguity": ambiguity,
        }
        audit_rows.append(_make_audit(source, aggregate, layers, view))

    width = max(row.shape[1] for row in audit_rows)
    padded_rows = [
        cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(250, 250, 250))
        for row in audit_rows
    ]
    audit_path = output_dir / "ring01_prong_instance_audit.png"
    cv2.imwrite(str(audit_path), cv2.vconcat(padded_rows))
    checks = {
        "all_proposal_unions_match_aggregate": all(item["aggregate_union_iou"] == 1.0 for item in view_reports.values()),
        "all_proposal_layers_are_disjoint": all(item["pairwise_overlap_px2"] == 0 for item in view_reports.values()),
        "three_views_have_four_resolved_instances": sum(item["resolved_individual_instances"] for item in view_reports.values()) == 3,
        "ambiguous_views_are_explicit": all(view_reports[view]["ambiguity"] for view in ("side", "back")),
        "all_four_stable_ids_present_in_every_view": all(set(item["component_observations"]) == set(COMPONENT_IDS) for item in view_reports.values()),
    }
    report = {
        "schema_version": "lalitha_phase4_prong_instance_evidence_v1",
        "object_id": "ring01",
        "execution_profile": "images_only",
        "authority": "machine_proposal_review_required",
        "component_ids": list(COMPONENT_IDS),
        "views": view_reports,
        "summary": {
            "view_count": len(VIEWS),
            "resolved_instance_view_count": len(RESOLVED_VIEWS),
            "ambiguous_projection_view_count": len(VIEWS) - len(RESOLVED_VIEWS),
            "resolved_views": [view for view in VIEWS if view in RESOLVED_VIEWS],
            "ambiguous_views": [view for view in VIEWS if view not in RESOLVED_VIEWS],
            "reviewed_instance_count": 0,
        },
        "checks": checks,
        "gate": {
            "partial_instance_evidence_available": all(checks.values()),
            "all_views_individually_resolved": False,
            "cross_view_identity_reviewed": False,
            "geometry_generation_authorized": False,
        },
        "artifacts": {"audit": _relative(audit_path, root)},
        "claims": {
            "individual_prong_segmentation_validated": False,
            "hidden_prong_identity_recovered": False,
            "cross_view_identity_validated": False,
        },
        "limitations": [
            "These masks are deterministic machine proposals, not human ground truth.",
            "Side and back evidence collapses prong pairs into unresolved projection groups.",
            "The angled identity ordering is a reviewable hypothesis, not calibrated camera proof.",
            "Image-plane masks do not establish hidden 3D prong geometry or millimetres.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase4_prong_instances_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = run(root, output_dir)
    print(json.dumps({"summary": report["summary"], "checks": report["checks"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
