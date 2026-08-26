"""Validation for the Phase 3.1 render-and-compare refinement loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import trimesh

from refine_phase3_1 import CameraHypothesis, _bbox, _comparison, _iou, _jewelry_transform, _read_mask, _render


VIEWS = ("front", "side", "top", "angled", "back")


def validate(base: Path) -> dict:
    fit = json.loads((base / "phase3_1_fit.json").read_text(encoding="utf-8"))
    semantic = json.loads((base / "semantic" / "phase3_semantic_proxy.json").read_text(encoding="utf-8"))
    before = {view: fit["initial"]["view_component_iou"][view]["jewelry"] for view in VIEWS}
    after = {view: fit["optimized"]["view_component_iou"][view]["jewelry"] for view in VIEWS}
    exported_mesh = trimesh.load(base / "semantic" / "ring01_phase3_semantic.stl", force="mesh")
    phase2_2_dir = Path(fit["source_masks"]).parent
    input_dir = phase2_2_dir.parent / "ring01_reference_images"
    exported_iou = {}
    exported_renders = {}
    for view in VIEWS:
        target = _read_mask(phase2_2_dir, view, "jewelry")
        frame = _bbox(target)
        camera = CameraHypothesis(**fit["camera_fit"]["cameras"][view])
        transform = _jewelry_transform(exported_mesh, camera, frame)
        rendered, _ = _render(exported_mesh, camera, target.shape, transform)
        exported_iou[view] = round(_iou(target, rendered), 5)
        render_path = base / "renders" / "exported_cad" / f"ring01_{view}_jewelry.png"
        render_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(render_path), rendered)
        exported_renders[view] = str(render_path)
        image = cv2.imread(str(input_dir / f"ring01_{view}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(input_dir / f"ring01_{view}.png")
        comparison = _comparison(image, target, rendered, f"{view.upper()} EXPORTED CAD IoU {exported_iou[view]:.3f}")
        comparison_path = base / "comparisons" / f"ring01_{view}_exported_cad.png"
        cv2.imwrite(str(comparison_path), comparison)
    checks = {
        "objective_improved_pass": fit["optimized"]["score"] > fit["initial"]["score"] + 0.005,
        "all_view_jewelry_iou_nonregression_pass": all(after[view] >= before[view] - 0.005 for view in VIEWS),
        "all_view_jewelry_iou_minimum_pass": all(after[view] >= 0.48 for view in VIEWS),
        "metal_single_solid_pass": semantic["validation"]["metal_is_single_solid"],
        "stone_single_solid_pass": semantic["validation"]["stone_is_single_solid"],
        "four_prongs_pass": semantic["topology"]["prong_count"] == 4,
        "cavity_pass": semantic["topology"]["boolean_cavity"],
        "exported_cad_all_view_iou_pass": all(exported_iou[view] >= 0.45 for view in VIEWS),
        "exported_cad_matches_proxy_pass": all(exported_iou[view] >= after[view] - 0.08 for view in VIEWS),
    }
    report = {
        "stage": "phase3_1_refinement_validation",
        "passed": all(checks.values()),
        "manufacturing_accuracy_validated": False,
        "checks": checks,
        "objective": {
            "initial": fit["initial"]["score"],
            "optimized": fit["optimized"]["score"],
            "absolute_improvement": fit["improvement"],
            "relative_improvement_percent": round(fit["improvement"] / max(1e-9, fit["initial"]["score"]) * 100, 3),
        },
        "jewelry_iou": {view: {"initial": before[view], "optimized": after[view], "delta": round(after[view] - before[view], 5)} for view in VIEWS},
        "exported_cad_jewelry_iou": exported_iou,
        "exported_cad_renders": exported_renders,
        "remaining_gate": "Visual inspection and metric calibration remain required; image-space fitting does not establish millimetre accuracy.",
    }
    for view in VIEWS:
        comparison = base / "comparisons" / f"ring01_{view}_phase3_1_before_after.png"
        if cv2.imread(str(comparison), cv2.IMREAD_COLOR) is None:
            raise FileNotFoundError(comparison)
    (base / "phase3_1_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-1-dir", type=Path, default=Path("data/ring01_phase3_1"))
    args = parser.parse_args()
    print(json.dumps(validate(args.phase3_1_dir), indent=2))


if __name__ == "__main__":
    main()
