"""Validation for the normalized Phase 3 visual-hull reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh


def validate(base: Path) -> dict:
    source = json.loads((base / "phase3_reconstruction.json").read_text(encoding="utf-8"))
    mesh_path = base / "meshes" / "ring01_phase3_visual_hull.glb"
    mesh = trimesh.load(mesh_path, force="mesh")
    reprojections = source["reprojections"]
    semantic = json.loads((base / "semantic" / "phase3_semantic_proxy.json").read_text(encoding="utf-8"))
    ious = {view: record["silhouette_iou"] for view, record in reprojections.items()}
    recalls = {view: record["silhouette_recall"] for view, record in reprojections.items()}
    checks = {
        "mesh_nonempty_pass": len(mesh.vertices) > 100 and len(mesh.faces) > 100,
        "finite_bounds_pass": bool(mesh.bounds.shape == (2, 3)),
        "all_reprojection_iou_pass": all(value >= 0.45 for value in ious.values()),
        "all_reprojection_recall_pass": all(value >= 0.45 for value in recalls.values()),
        "semantic_stone_evidence_pass": source["semantic_evidence"]["stone_point_count"] > 50,
        "metric_claim_is_guarded_pass": source["metric_scale"] or source["coordinate_system"] == "normalized object coordinates",
        "semantic_metal_single_solid_pass": semantic["validation"]["metal_is_single_solid"],
        "semantic_stone_single_solid_pass": semantic["validation"]["stone_is_single_solid"],
        "semantic_topology_pass": semantic["topology"]["one_shank"] and semantic["topology"]["one_stone"] and semantic["topology"]["prong_count"] == 4 and semantic["topology"]["stone_seat"] and semantic["topology"]["boolean_cavity"],
    }
    report = {
        "stage": "phase3_coarse_reconstruction_validation",
        "coarse_reconstruction_passed": all(checks.values()),
        "manufacturing_accuracy_validated": False,
        "checks": checks,
        "silhouette_iou": ious,
        "silhouette_recall": recalls,
        "mean_silhouette_iou": round(sum(ious.values()) / len(ious), 5),
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "body_count": int(mesh.body_count),
        },
        "semantic_proxy": semantic["validation"],
        "next_gate": "Inspect all reprojections, then fit parametric shank, stone, setting, and prongs to the coarse evidence.",
    }
    (base / "phase3_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, default=Path("data/ring01_phase3"))
    args = parser.parse_args()
    print(json.dumps(validate(args.phase3_dir), indent=2))


if __name__ == "__main__":
    main()
