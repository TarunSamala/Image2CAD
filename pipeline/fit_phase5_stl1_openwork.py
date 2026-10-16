"""Trial exact front-view negative-space cuts on STL-1 Phase 5.2 rings.

Every Phase 4 front-view negative-space candidate is converted into an exact
Boolean cutter.  A cutter is retained only when it intersects the head and the
result remains one valid connected head solid.  Candidate semantics remain
unresolved: this stage models observable open space, not stones or materials.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.exact_cad_v2 import (
    HeadSupportSpec,
    RingBodySpec,
    build_head_support_envelope,
    build_profiled_ring_body,
)
from pipeline.fit_phase5_stl1_head_supports import (
    _bbox,
    _front_profile,
    _head_dimensions,
    _largest_contour,
    _overlay,
    _render_front_profile,
    _render_geometry_preview,
    _sample_closed_contour,
    _top_profile,
)
from pipeline.fit_phase5_stl1_ring_bodies import (
    NOMINAL_INNER_DIAMETER,
    _boundary_f1,
    _iou,
    _profile_controls,
)


SCHEMA_VERSION = "lalitha_phase5_stl1_exact_openwork_fit_v1"


def _component_profile(
    component_mask: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    width: float,
    depth: float,
) -> tuple[tuple[float, float], ...]:
    """Map a view-local negative-space contour into the front X-Y plane."""
    contour = _sample_closed_contour(_largest_contour(component_mask), 48)
    x0, y0, x1, y1 = target_bbox
    center_x = (x0 + x1 - 1) / 2.0
    center_y = (y0 + y1 - 1) / 2.0
    scale_x = width / max(x1 - x0, 1)
    scale_y = depth / max(y1 - y0, 1)
    return tuple(
        ((float(x) - center_x) * scale_x, (float(y) - center_y) * scale_y)
        for x, y in contour
    )


def _render_openwork(
    outer_profile: tuple[tuple[float, float], ...],
    opening_profiles: list[tuple[tuple[float, float], ...]],
    shape: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    width: float,
    depth: float,
) -> np.ndarray:
    rendered = _render_front_profile(
        outer_profile, shape, target_bbox, width, depth
    )
    x0, y0, x1, y1 = target_bbox
    center_x = (x0 + x1 - 1) / 2.0
    center_y = (y0 + y1 - 1) / 2.0
    scale_x = (x1 - x0) / width
    scale_y = (y1 - y0) / depth
    for profile in opening_profiles:
        polygon = np.rint(
            np.asarray(
                [
                    (center_x + x * scale_x, center_y + y * scale_y)
                    for x, y in profile
                ]
            )
        ).astype(np.int32)
        cv2.fillPoly(rendered, [polygon], 0)
    return rendered


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item["metrics"][key] for item in items if item["metrics"]]
    return round(float(np.mean(values)), 6) if values else None


def _render_audit(
    root: Path, objects: list[dict[str, Any]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 4, figsize=(14, 19), facecolor="#f4f4f4")
    for axis, item in zip(axes.flat, objects):
        source = cv2.cvtColor(
            cv2.imread(str(root / item["source_front_image"])),
            cv2.COLOR_BGR2RGB,
        )
        if item["evaluation_status"] == "held_out_not_scored":
            axis.imshow(source, alpha=0.42)
            axis.text(
                0.5,
                0.5,
                "HELD OUT\nNOT SCORED",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            detail = (
                f"{item['accepted_opening_count']} / "
                f"{item['input_candidate_count']} cuts retained"
            )
        else:
            target = cv2.imread(
                str(root / item["artifacts"]["front_target"]),
                cv2.IMREAD_GRAYSCALE,
            )
            rendered = cv2.imread(
                str(root / item["artifacts"]["front_render"]),
                cv2.IMREAD_GRAYSCALE,
            )
            axis.imshow(_overlay(source, target, rendered))
            detail = (
                f"IoU {item['metrics']['openwork_iou']:.3f} · "
                f"delta {item['metrics']['iou_delta']:+.3f} · "
                f"cuts {item['accepted_opening_count']}/"
                f"{item['input_candidate_count']}"
            )
        axis.set_title(
            f"{item['object_id'].upper()} · {item['split'].upper()}\n{detail}",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.set_axis_off()
    figure.suptitle(
        "PHASE 5.3 · EXACT IMAGE-DERIVED OPENWORK TRIALS",
        fontsize=17,
        fontweight="bold",
        color="#20252b",
    )
    figure.text(
        0.5,
        0.012,
        "Pink = machine target · white = retained exact-cut render · candidate semantics unresolved",
        ha="center",
        fontsize=10.5,
        color="#4a5159",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.965))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    import cadquery as cq

    parameter_report = json.loads(
        (root / "data/phase5_stl1_ring_parameters_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    body_report = json.loads(
        (root / "data/phase5_stl1_exact_ring_body_fit_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_report = json.loads(
        (root / "data/phase4_stl1_component_candidates_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    body_by_id = {item["object_id"]: item for item in body_report["objects"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []

    for source in parameter_report["objects"]:
        object_id = source["object_id"]
        ring_spec = RingBodySpec(**body_by_id[object_id]["ring_body_spec"])
        inner_profile, outer_profile = _profile_controls(
            source["ring_parameter_evidence"],
            source["normalized_parameter_proposal"]["shank"],
        )
        ring_body = build_profiled_ring_body(
            cq, ring_spec, inner_profile, outer_profile
        )
        dimensions = _head_dimensions(source, root, ring_spec)
        front_mask = cv2.imread(
            str(root / source["views"]["front"]["mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        ) > 127
        front_bbox = _bbox(front_mask)
        front_profile = _front_profile(
            front_mask, dimensions["head_width"], dimensions["head_depth"]
        )
        support_spec = HeadSupportSpec(
            component_id=f"{object_id}_head_support_001",
            height=dimensions["head_height"],
            base_z=ring_spec.inner_height / 2.0
            + ring_spec.radial_thickness * 0.15,
            shoulder_radius=max(0.35, ring_spec.radial_thickness * 0.62),
        )
        top_mask = cv2.imread(
            str(root / source["views"]["top"]["mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        ) > 127
        start_y = int(dimensions["top_start_y_px"])
        top_target = np.zeros_like(top_mask, dtype=np.uint8)
        top_target[start_y:] = top_mask[start_y:].astype(np.uint8) * 255
        diameter_px = float(
            source["ring_parameter_evidence"]["aperture"][
                "equivalent_diameter_px"
            ]
        )
        units_per_pixel = NOMINAL_INNER_DIAMETER / diameter_px
        top_profile = _top_profile(
            top_target > 0,
            dimensions["top_center_x_px"],
            start_y,
            units_per_pixel,
            support_spec.base_z,
        )

        source_candidates = candidate_report["objects"][object_id]["views"][
            "front"
        ]["candidates"]
        candidate_profiles: list[tuple[tuple[float, float], ...]] = []
        for candidate in source_candidates:
            candidate_mask = cv2.imread(
                str(root / candidate["mask_path"]), cv2.IMREAD_GRAYSCALE
            ) > 127
            candidate_profiles.append(
                _component_profile(
                    candidate_mask,
                    front_bbox,
                    dimensions["head_width"],
                    dimensions["head_depth"],
                )
            )

        support = build_head_support_envelope(
            cq,
            ring_body,
            support_spec,
            front_profile,
            top_profile,
            tuple(candidate_profiles),
        )
        opening_records = []
        accepted_profiles = []
        for candidate, profile, trial in zip(
            source_candidates, candidate_profiles, support["opening_trials"]
        ):
            if trial["accepted"]:
                accepted_profiles.append(profile)
            opening_records.append(
                {
                    "component_id": candidate["candidate_id"],
                    "candidate_subtype": candidate["candidate_subtype"],
                    "semantic_state": candidate["semantic_state"],
                    "review_state": candidate["review_state"],
                    "area_px2": candidate["area_px2"],
                    "accepted_as_exact_cut_candidate": trial["accepted"],
                    "cut_status": trial["reason"],
                    "removed_volume_nominal_units3": round(
                        float(trial["removed_volume"]), 8
                    ),
                }
            )

        front_target = front_mask.astype(np.uint8) * 255
        baseline_render = _render_front_profile(
            front_profile,
            front_mask.shape,
            front_bbox,
            dimensions["head_width"],
            dimensions["head_depth"],
        )
        front_render = _render_openwork(
            front_profile,
            accepted_profiles,
            front_mask.shape,
            front_bbox,
            dimensions["head_width"],
            dimensions["head_depth"],
        )

        object_dir = output_dir / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "front_target": object_dir / "front_openwork_target.png",
            "front_baseline": object_dir / "front_filled_baseline.png",
            "front_render": object_dir / "front_exact_openwork_render.png",
            "step": object_dir / f"{object_id}_shank_head_openwork.step",
        }
        cv2.imwrite(str(paths["front_target"]), front_target)
        cv2.imwrite(str(paths["front_baseline"]), baseline_render)
        cv2.imwrite(str(paths["front_render"]), front_render)
        cq.exporters.export(cq.Workplane(obj=support["metal"]), str(paths["step"]))
        if source["split"] == "val":
            paths["geometry_preview"] = (
                object_dir / f"{object_id}_openwork_geometry_preview.png"
            )
            _render_geometry_preview(
                support["metal"],
                paths["geometry_preview"],
                f"PHASE 5.3 · {object_id.upper()} OPENWORK CANDIDATE",
                "Actual tessellated STEP candidate · retained openings included · relief excluded",
            )

        scored = source["split"] != "test"
        metrics = None
        if scored:
            baseline_iou = _iou(front_target, baseline_render)
            openwork_iou = _iou(front_target, front_render)
            metrics = {
                "filled_baseline_iou": round(baseline_iou, 6),
                "openwork_iou": round(openwork_iou, 6),
                "iou_delta": round(openwork_iou - baseline_iou, 6),
                "openwork_boundary_f1": round(
                    _boundary_f1(front_target, front_render), 6
                ),
            }
        accepted_count = sum(
            item["accepted_as_exact_cut_candidate"] for item in opening_records
        )
        objects.append(
            {
                "object_id": object_id,
                "category": "ring",
                "split": source["split"],
                "evaluation_status": (
                    "scored" if scored else "held_out_not_scored"
                ),
                "source_front_image": source["views"]["front"]["image_path"],
                "ring_body_spec": asdict(ring_spec),
                "head_support_spec": asdict(support_spec),
                "input_candidate_count": len(opening_records),
                "accepted_opening_count": accepted_count,
                "rejected_opening_count": len(opening_records) - accepted_count,
                "opening_candidates": opening_records,
                "exact_geometry": {
                    "head_support_valid": bool(
                        support["head_support"].isValid()
                    ),
                    "head_support_solid_count": len(
                        support["head_support"].Solids()
                    ),
                    "metal_valid": bool(support["metal"].isValid()),
                    "metal_solid_count": len(support["metal"].Solids()),
                },
                "metrics": metrics,
                "artifacts": {
                    key: _relative(path, root) for key, path in paths.items()
                },
            }
        )

    audit_path = output_dir / "stl1_exact_openwork_fit_audit.png"
    _render_audit(root, objects, audit_path)
    train = [item for item in objects if item["split"] == "train"]
    validation = [item for item in objects if item["split"] == "val"]
    test = [item for item in objects if item["split"] == "test"]
    input_count = sum(item["input_candidate_count"] for item in objects)
    accepted_count = sum(item["accepted_opening_count"] for item in objects)
    rejected_count = sum(item["rejected_opening_count"] for item in objects)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_profile": "ring_only_front_negative_space_exact_cut_trials",
        "scope": {
            "included": [
                "Phase 5.2 exact shank and head-support envelope",
                "Phase 4 front-view negative-space candidates",
                "valid connected exact Boolean cut trials",
            ],
            "excluded": [
                "semantic stone classification",
                "relief or face reconstruction",
                "cross-view opening identity",
                "manufacturing scale",
            ],
        },
        "summary": {
            "object_count": len(objects),
            "input_candidate_count": input_count,
            "accepted_opening_count": accepted_count,
            "rejected_opening_count": rejected_count,
            "train_mean_filled_baseline_iou": _mean(
                train, "filled_baseline_iou"
            ),
            "train_mean_openwork_iou": _mean(train, "openwork_iou"),
            "train_mean_iou_delta": _mean(train, "iou_delta"),
            "validation_mean_filled_baseline_iou": _mean(
                validation, "filled_baseline_iou"
            ),
            "validation_mean_openwork_iou": _mean(
                validation, "openwork_iou"
            ),
            "validation_mean_iou_delta": _mean(validation, "iou_delta"),
            "validation_mean_openwork_boundary_f1": _mean(
                validation, "openwork_boundary_f1"
            ),
            "test_held_out_count": len(test),
        },
        "checks": {
            "all_24_ring_candidates_generated": len(objects) == 24,
            "all_front_candidates_trialed": (
                input_count == accepted_count + rejected_count
            ),
            "all_heads_valid_single_solids": all(
                item["exact_geometry"]["head_support_valid"]
                and item["exact_geometry"]["head_support_solid_count"] == 1
                for item in objects
            ),
            "all_metals_valid_single_solids": all(
                item["exact_geometry"]["metal_valid"]
                and item["exact_geometry"]["metal_solid_count"] == 1
                for item in objects
            ),
            "all_step_exports_present": all(
                (root / item["artifacts"]["step"]).is_file()
                for item in objects
            ),
            "all_validation_geometry_previews_present": all(
                (root / item["artifacts"]["geometry_preview"]).is_file()
                for item in validation
            ),
            "held_out_test_not_scored": len(test) == 3
            and all(item["metrics"] is None for item in test),
            "candidate_semantics_not_promoted": all(
                candidate["semantic_state"]
                == "observable_geometry_semantics_unresolved"
                for item in objects
                for candidate in item["opening_candidates"]
            ),
            "ring_only_scope_enforced": all(
                item["category"] == "ring" for item in objects
            ),
        },
        "objects": objects,
        "artifacts": {"audit": _relative(audit_path, root)},
        "decision": "exact_negative_space_cut_candidates_ready_for_review",
        "claims": {
            "observable_negative_space_converted_to_exact_cut_candidates": True,
            "openings_semantically_reviewed": False,
            "relief_reconstructed": False,
            "complete_ring_reconstructed": False,
            "manufacturing_accuracy_validated": False,
        },
        "limitations": [
            "Candidate masks and fitting scores come from the same machine-derived front silhouettes.",
            "A visible enclosed region can be openwork, a bezel interior, a missing insert or a segmentation error.",
            "The Boolean trials do not establish cross-view opening identity or hidden cavity depth.",
            "The exact cuts traverse the structural head envelope and remain review candidates, not approved manufacturing holes.",
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
        default=Path("data/phase5_stl1_exact_openwork_fit_v1"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    report = run(root, output_dir)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "summary": report["summary"],
                "checks": report["checks"],
                "artifacts": report["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
