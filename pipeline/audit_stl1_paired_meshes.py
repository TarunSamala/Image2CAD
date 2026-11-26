"""Audit the 24 online-generated STL-1 OBJ targets as paired pseudo-ground-truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["source_object_name"]: record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    geometries = [
        geometry
        for geometry in loaded.geometry.values()
        if len(geometry.vertices) and len(geometry.faces)
    ]
    if not geometries:
        raise ValueError(f"no triangle geometry in {path}")
    return geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)


def _declares_material(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return "mtllib" in handle.read(4096)


def audit(dataset_root: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    objects = []
    hashes: dict[str, list[str]] = {}
    for directory in sorted(dataset_root.glob("Ring *"), key=lambda item: int(item.name.split()[-1])):
        source = manifest.get(directory.name)
        if source is None:
            raise ValueError(f"{directory.name} is absent from the prepared manifest")
        obj_files = sorted(directory.glob("*.obj"))
        if len(obj_files) != 1:
            raise ValueError(f"{directory.name} must contain exactly one OBJ")
        path = obj_files[0]
        digest = _sha256(path)
        hashes.setdefault(digest, []).append(source["object_id"])
        mesh = _mesh(path)
        vertices = np.asarray(mesh.vertices)
        finite = bool(np.isfinite(vertices).all())
        component_count = int(mesh.body_count)
        objects.append(
            {
                "object_id": source["object_id"],
                "source_object_name": directory.name,
                "split": source["split"],
                "capture_tier": "A_exploratory",
                "target_class": "online_generated_paired_mesh_pseudo_ground_truth",
                "images": {
                    "count": len(source["views"]),
                    "views": sorted(source["views"]),
                    "all_source_paths_exist": all(
                        Path(record["source_path"]).is_file()
                        for record in source["views"].values()
                    ),
                },
                "mesh": {
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "finite_vertices": finite,
                    "watertight": bool(mesh.is_watertight),
                    "winding_consistent": bool(mesh.is_winding_consistent),
                    "is_volume": bool(mesh.is_volume),
                    "connected_components": component_count,
                    "euler_number": int(mesh.euler_number),
                    "bounds": np.round(mesh.bounds, 7).tolist(),
                    "extents": np.round(mesh.extents, 7).tolist(),
                    "units": mesh.units,
                    "material_file_declared": _declares_material(path),
                },
            }
        )

    duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    report = {
        "schema_version": "image2cad_stl1_paired_mesh_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "STL-1",
        "capture_tier": "A_exploratory",
        "object_count": len(objects),
        "mesh_count": len(objects),
        "view_count": sum(item["images"]["count"] for item in objects),
        "splits": {
            split: sum(item["split"] == split for item in objects)
            for split in ("train", "val", "test")
        },
        "checks": {
            "exactly_24_objects": len(objects) == 24,
            "one_mesh_per_object": len(objects) == len(hashes),
            "five_views_per_object": all(item["images"]["count"] == 5 for item in objects),
            "all_source_images_exist": all(
                item["images"]["all_source_paths_exist"] for item in objects
            ),
            "all_vertices_finite": all(item["mesh"]["finite_vertices"] for item in objects),
            "all_meshes_unique": not duplicates,
        },
        "mesh_quality": {
            "watertight_count": sum(item["mesh"]["watertight"] for item in objects),
            "volume_count": sum(item["mesh"]["is_volume"] for item in objects),
            "single_component_count": sum(
                item["mesh"]["connected_components"] == 1 for item in objects
            ),
            "material_file_count": sum(
                item["mesh"]["material_file_declared"] for item in objects
            ),
            "duplicate_groups": duplicates,
        },
        "claim_policy": {
            "allowed": [
                "paired image-to-mesh benchmarking",
                "normalized geometry comparison",
                "pseudo-ground-truth training experiments",
            ],
            "forbidden": [
                "manufacturing-accurate CAD ground truth",
                "physical scale without a known dimension",
                "hidden geometry correctness",
            ],
        },
        "objects": objects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/STL-1"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("dataset/prepared_v1/manifest.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/phase_runs/v1/stl1_paired_mesh_audit_v1/report.json"),
    )
    args = parser.parse_args()
    report = audit(args.dataset, args.manifest, args.output)
    print(json.dumps({
        "object_count": report["object_count"],
        "splits": report["splits"],
        "checks": report["checks"],
        "mesh_quality": report["mesh_quality"],
    }, indent=2))


if __name__ == "__main__":
    main()
