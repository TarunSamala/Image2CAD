"""Benchmark current STL-1 visual hulls against paired online-generated OBJs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from pipeline.run_saraswati_blind_benchmark import (
    _align,
    _load_mesh,
    _normalize,
    _point_preview,
    _sha256,
    _surface_metrics,
    _write_json,
)


def _manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["object_id"]: record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def _sample(mesh: trimesh.Trimesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    points, faces = trimesh.sample.sample_surface(mesh, count)
    normals = np.asarray(mesh.face_normals[faces], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return points, normals


def _normal_metrics(
    predicted_points: np.ndarray,
    predicted_normals: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
) -> dict[str, float]:
    target_tree = cKDTree(target_points)
    _, nearest = target_tree.query(predicted_points, k=1, workers=-1)
    cosine = np.abs(np.sum(predicted_normals * target_normals[nearest], axis=1))
    cosine = np.clip(cosine, 0, 1)
    return {
        "mean_absolute_cosine": round(float(np.mean(cosine)), 7),
        "median_absolute_cosine": round(float(np.median(cosine)), 7),
        "fraction_above_0_90": round(float(np.mean(cosine >= 0.90)), 7),
    }


def _mean_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    return {
        "object_count": len(items),
        "chamfer_l1_normalized": round(float(np.mean([
            item["surface_metrics"]["chamfer_l1_normalized"] for item in items
        ])), 8),
        "hausdorff_95_normalized": round(float(np.mean([
            item["surface_metrics"]["hausdorff_95_normalized"] for item in items
        ])), 8),
        "f_score_1pct": round(float(np.mean([
            item["surface_metrics"]["surface_f_scores"]["1pct"]["f_score"] for item in items
        ])), 7),
        "f_score_2pct": round(float(np.mean([
            item["surface_metrics"]["surface_f_scores"]["2pct"]["f_score"] for item in items
        ])), 7),
        "f_score_5pct": round(float(np.mean([
            item["surface_metrics"]["surface_f_scores"]["5pct"]["f_score"] for item in items
        ])), 7),
        "normal_mean_absolute_cosine": round(float(np.mean([
            item["normal_consistency"]["mean_absolute_cosine"] for item in items
        ])), 7),
    }


def benchmark(
    root: Path,
    manifest_path: Path,
    phase3_report_path: Path,
    output_root: Path,
    sample_count: int = 50_000,
) -> dict[str, Any]:
    records = _manifest(manifest_path)
    phase3 = json.loads(phase3_report_path.read_text(encoding="utf-8"))
    phase3_objects = {item["object_id"]: item for item in phase3["objects"]}
    output_root.mkdir(parents=True, exist_ok=True)
    objects = []
    for ordinal, object_id in enumerate(sorted(records)):
        record = records[object_id]
        target_candidates = list((root / "dataset/STL-1" / record["source_object_name"]).glob("*.obj"))
        if len(target_candidates) != 1:
            raise ValueError(f"expected one paired OBJ for {object_id}")
        proposal_path = root / phase3_objects[object_id]["exports"]["stl"]
        target_path = target_candidates[0]
        proposal = _load_mesh(proposal_path)
        target = _load_mesh(target_path)
        proposal_points, proposal_normals = _sample(proposal, sample_count, 1700 + ordinal)
        target_points, target_normals = _sample(target, sample_count, 2700 + ordinal)
        proposal_points, _, _ = _normalize(proposal_points)
        target_points, _, _ = _normalize(target_points)
        aligned, alignment = _align(proposal_points, target_points)
        rotation = np.asarray(alignment["rotation_source_to_target"], dtype=np.float64)
        aligned_normals = proposal_normals @ rotation.T
        metrics = _surface_metrics(aligned, target_points)
        normals = _normal_metrics(aligned, aligned_normals, target_points, target_normals)
        object_root = output_root / object_id
        object_root.mkdir(exist_ok=True)
        preview_path = object_root / "aligned_surface_comparison.png"
        _point_preview(aligned, target_points, preview_path)
        result = {
            "object_id": object_id,
            "split": record["split"],
            "proposal": {
                "path": str(proposal_path.relative_to(root)),
                "sha256": _sha256(proposal_path),
                "source": "phase3_non_metric_visual_hull",
            },
            "paired_target": {
                "path": str(target_path.relative_to(root)),
                "sha256": _sha256(target_path),
                "class": "online_generated_mesh_pseudo_ground_truth",
            },
            "sampling": {"points_per_mesh": sample_count},
            "alignment": alignment,
            "surface_metrics": metrics,
            "normal_consistency": normals,
            "preview": str(preview_path.relative_to(root)),
            "metric_scale_validated": False,
            "manufacturing_accuracy_validated": False,
        }
        _write_json(object_root / "report.json", result)
        objects.append(result)

    split_metrics = {
        split: _mean_metrics([item for item in objects if item["split"] == split])
        for split in ("train", "val", "test")
    }
    report = {
        "schema_version": "image2cad_stl1_paired_mesh_benchmark_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "post_hoc_paired_mesh_evaluation",
        "object_count": len(objects),
        "sampling_points_per_mesh": sample_count,
        "alignment": "independent_normalization_plus_24_axis_trimmed_ICP",
        "overall": _mean_metrics(objects),
        "splits": split_metrics,
        "test_policy": {
            "test_object_count": split_metrics["test"]["object_count"],
            "used_for_parameter_selection": False,
            "result_is_evaluation_only": True,
        },
        "claims": {
            "measures_normalized_surface_similarity": True,
            "uses_exact_CAD_ground_truth": False,
            "metric_scale_validated": False,
            "topology_validated": False,
            "manufacturing_accuracy_validated": False,
        },
        "objects": objects,
    }
    _write_json(output_root / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/prepared_v1/manifest.jsonl"))
    parser.add_argument(
        "--phase3-report",
        type=Path,
        default=Path("dataset/phase_runs/v1/phase3/visual_hull/phase3_batch_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/phase_runs/v1/stl1_paired_mesh_benchmark_v1"),
    )
    parser.add_argument("--samples", type=int, default=50_000)
    args = parser.parse_args()
    root = args.root.resolve()
    report = benchmark(
        root,
        root / args.manifest,
        root / args.phase3_report,
        root / args.output,
        args.samples,
    )
    print(json.dumps({
        "object_count": report["object_count"],
        "overall": report["overall"],
        "splits": report["splits"],
        "test_policy": report["test_policy"],
    }, indent=2))


if __name__ == "__main__":
    main()
