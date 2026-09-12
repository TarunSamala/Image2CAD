"""Plan and validate isolated 2D-to-3D model experiments.

The benchmark never executes commands from the registry. External generators
run in isolated environments and publish a small manifest; this module then
loads their output independently and applies one common validation contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_REGISTRY = Path("benchmarks/2d_to_3d/models.json")


@dataclass(frozen=True)
class Hardware:
    gpu_name: str | None
    gpu_total_gb: float
    gpu_free_gb: float
    ram_total_gb: float
    disk_free_gb: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = _read_json(path)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema: {registry.get('schema_version')!r}")
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("registry must contain a non-empty models list")
    required = {
        "id", "name", "official_repo", "license", "license_class",
        "license_review_required", "input_modes", "output_types",
        "min_vram_gb", "cpu_supported", "adapter", "role", "limitations",
    }
    ids: set[str] = set()
    for index, model in enumerate(models):
        missing = required - set(model)
        if missing:
            raise ValueError(f"model {index} is missing fields: {sorted(missing)}")
        if model["id"] in ids:
            raise ValueError(f"duplicate model id: {model['id']}")
        ids.add(model["id"])
        if model["adapter"] not in {"existing_pipeline", "external_import"}:
            raise ValueError(f"unsafe or unknown adapter for {model['id']}: {model['adapter']}")
    return registry


def _gpu_info() -> tuple[str | None, float, float]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        first = result.stdout.strip().splitlines()[0]
        name, total_mib, free_mib = [part.strip() for part in first.rsplit(",", 2)]
        return name, float(total_mib) / 1024.0, float(free_mib) / 1024.0
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None, 0.0, 0.0


def detect_hardware(output_root: Path = Path("runs")) -> Hardware:
    gpu_name, gpu_total, gpu_free = _gpu_info()
    page_size = int(getattr(__import__("os"), "sysconf")("SC_PAGE_SIZE"))
    pages = int(getattr(__import__("os"), "sysconf")("SC_PHYS_PAGES"))
    existing = output_root if output_root.exists() else output_root.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return Hardware(
        gpu_name=gpu_name,
        gpu_total_gb=round(gpu_total, 3),
        gpu_free_gb=round(gpu_free, 3),
        ram_total_gb=round(page_size * pages / 1024**3, 3),
        disk_free_gb=round(shutil.disk_usage(existing).free / 1024**3, 3),
    )


def classify(model: dict[str, Any], hardware: Hardware) -> tuple[str, list[str]]:
    reasons: list[str] = []
    required = model["min_vram_gb"]
    if model["adapter"] == "existing_pipeline":
        return "local_ready", ["implemented in the existing pipeline"]
    if model["license_review_required"]:
        reasons.append("license/checkpoint review required")
    if required is None:
        reasons.append("official minimum VRAM is not pinned in the registry")
        if model["cpu_supported"]:
            return "manual_audit_or_cpu", reasons
        return "remote_gpu_after_audit", reasons
    if required <= hardware.gpu_total_gb:
        if required > hardware.gpu_free_gb:
            reasons.append(f"needs {required:g} GB VRAM; only {hardware.gpu_free_gb:g} GB is currently free")
            return "local_after_memory_cleanup", reasons
        reasons.append(f"documented minimum {required:g} GB fits the detected GPU")
        return "local_candidate", reasons
    reasons.append(f"needs about {required:g} GB VRAM; detected GPU has {hardware.gpu_total_gb:g} GB")
    if model["cpu_supported"]:
        reasons.append("CPU fallback exists but may be impractically slow")
        return "cpu_fallback_or_remote", reasons
    return "remote_gpu_required", reasons


def build_plan(registry: dict[str, Any], hardware: Hardware) -> dict[str, Any]:
    planned = []
    for model in registry["models"]:
        status, reasons = classify(model, hardware)
        planned.append(
            {
                "id": model["id"],
                "name": model["name"],
                "status": status,
                "reasons": reasons,
                "input_modes": model["input_modes"],
                "output_types": model["output_types"],
                "minimum_vram_gb": model["min_vram_gb"],
                "license": model["license"],
                "license_review_required": model["license_review_required"],
                "official_repo": model["official_repo"],
            }
        )
    counts: dict[str, int] = {}
    for item in planned:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "hardware": asdict(hardware),
        "policy": registry["policy"],
        "model_count": len(planned),
        "status_counts": counts,
        "models": planned,
    }


def _model(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    matches = [model for model in registry["models"] if model["id"] == model_id]
    if not matches:
        raise ValueError(f"model is not registered: {model_id}")
    return matches[0]


def _load_mesh(path: Path):
    try:
        import numpy as np
        import trimesh
    except ImportError as error:
        raise RuntimeError("mesh validation requires numpy and trimesh") from error
    loaded = trimesh.load(path, force="scene")
    geometries = [geometry for geometry in loaded.geometry.values() if len(geometry.vertices) and len(geometry.faces)]
    if not geometries:
        raise ValueError(f"no triangle geometry found in {path}")
    mesh = trimesh.util.concatenate(geometries)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    finite = bool(np.isfinite(vertices).all())
    nondegenerate = mesh.nondegenerate_faces() if hasattr(mesh, "nondegenerate_faces") else np.ones(len(faces), dtype=bool)
    bodies = mesh.split(only_watertight=False)
    extents = [float(value) for value in mesh.extents]
    return mesh, {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "connected_bodies": int(len(bodies)),
        "finite_vertices": finite,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "degenerate_faces": int(len(faces) - int(np.count_nonzero(nondegenerate))),
        "bounds": [[float(value) for value in row] for row in mesh.bounds],
        "extents": extents,
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }


def _mask_metrics(target_path: Path, rendered_path: Path) -> dict[str, float]:
    try:
        import cv2
        import numpy as np
        from phase3_objective import boundary_f1, iou
    except ImportError as error:
        raise RuntimeError("mask validation requires OpenCV, numpy and phase3_objective") from error
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    rendered = cv2.imread(str(rendered_path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(target_path)
    if rendered is None:
        raise FileNotFoundError(rendered_path)
    if target.shape != rendered.shape:
        raise ValueError(f"mask shape mismatch: {target.shape} != {rendered.shape}")
    target = np.where(target > 127, 255, 0).astype(np.uint8)
    rendered = np.where(rendered > 127, 255, 0).astype(np.uint8)
    return {
        "silhouette_iou": round(iou(target, rendered), 6),
        "external_boundary_f1_2px": round(boundary_f1(target, rendered, tolerance=2, external_only=True), 6),
    }


def validate_manifest(manifest_path: Path, registry_path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    registry = load_registry(registry_path)
    model = _model(registry, manifest.get("model_id", ""))
    mesh_path = Path(manifest["mesh_path"])
    if not mesh_path.is_absolute():
        mesh_path = (manifest_path.parent / mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh, mesh_metrics = _load_mesh(mesh_path)

    target_masks = manifest.get("target_masks", {})
    rendered_masks = manifest.get("rendered_masks", {})
    if set(target_masks) != set(rendered_masks):
        raise ValueError("target_masks and rendered_masks must have exactly the same view names")
    view_metrics: dict[str, dict[str, float]] = {}
    for view in sorted(target_masks):
        target = Path(target_masks[view])
        rendered = Path(rendered_masks[view])
        if not target.is_absolute():
            target = (manifest_path.parent / target).resolve()
        if not rendered.is_absolute():
            rendered = (manifest_path.parent / rendered).resolve()
        view_metrics[view] = _mask_metrics(target, rendered)

    mean_iou = None
    mean_boundary = None
    floor_iou = None
    if view_metrics:
        count = len(view_metrics)
        mean_iou = sum(item["silhouette_iou"] for item in view_metrics.values()) / count
        mean_boundary = sum(item["external_boundary_f1_2px"] for item in view_metrics.values()) / count
        floor_iou = min(item["silhouette_iou"] for item in view_metrics.values())

    target_height = manifest.get("target_height_mm")
    source_height = mesh_metrics["extents"][2]
    scale = None
    if target_height is not None:
        if float(target_height) <= 0 or source_height <= 0:
            raise ValueError("target height and mesh Z extent must be positive")
        scale = float(target_height) / source_height

    ground_truth_kind = manifest.get("ground_truth_kind", "none")
    human_ground_truth = ground_truth_kind == "human_reviewed"
    geometry_checks = {
        "mesh_nonempty": mesh_metrics["vertices"] > 0 and mesh_metrics["faces"] > 0,
        "finite_vertices": mesh_metrics["finite_vertices"],
        "no_degenerate_faces": mesh_metrics["degenerate_faces"] == 0,
        "watertight": mesh_metrics["watertight"],
        "winding_consistent": mesh_metrics["winding_consistent"],
    }
    numerical_gate = bool(
        view_metrics
        and mean_iou is not None and mean_iou >= 0.80
        and floor_iou is not None and floor_iou >= 0.70
        and mean_boundary is not None and mean_boundary >= 0.75
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": model["id"],
            "name": model["name"],
            "official_repo": model["official_repo"],
            "license": model["license"],
            "license_review_required": model["license_review_required"],
        },
        "input_mode": manifest.get("input_mode", "unspecified"),
        "source_images": manifest.get("source_images", {}),
        "mesh": {
            "path": str(mesh_path),
            "sha256": _sha256(mesh_path),
            "bytes": mesh_path.stat().st_size,
            **mesh_metrics,
        },
        "scale": {
            "target_height_mm": target_height,
            "source_height_model_units": source_height,
            "uniform_scale_factor": scale,
            "metric_calibrated": bool(manifest.get("metric_calibrated", False)),
        },
        "ground_truth_kind": ground_truth_kind,
        "human_ground_truth": human_ground_truth,
        "views": view_metrics,
        "summary": {
            "mean_silhouette_iou": round(mean_iou, 6) if mean_iou is not None else None,
            "principal_view_floor": round(floor_iou, 6) if floor_iou is not None else None,
            "mean_external_boundary_f1_2px": round(mean_boundary, 6) if mean_boundary is not None else None,
            "numerical_visual_gate_passed": numerical_gate,
        },
        "geometry_checks": geometry_checks,
        "validation": {
            "mesh_import_passed": all(geometry_checks[key] for key in ("mesh_nonempty", "finite_vertices")),
            "topology_passed": all(geometry_checks.values()),
            "visual_accuracy_validated": numerical_gate and human_ground_truth,
            "manufacturing_accuracy_validated": False,
        },
        "warnings": [
            "A generated mesh is a reconstruction proposal, not editable CAD or manufacturing truth.",
            "Scores against machine or pseudo masks do not validate visual accuracy."
            if not human_ground_truth else
            "Human masks validate supplied views only; hidden geometry remains unverified.",
        ],
    }
    return report


def rank_reports(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        report = _read_json(path)
        summary = report["summary"]
        rows.append(
            {
                "model_id": report["model"]["id"],
                "report": str(path),
                "mean_silhouette_iou": summary["mean_silhouette_iou"],
                "principal_view_floor": summary["principal_view_floor"],
                "mean_external_boundary_f1_2px": summary["mean_external_boundary_f1_2px"],
                "watertight": report["mesh"]["watertight"],
                "connected_bodies": report["mesh"]["connected_bodies"],
                "human_ground_truth": report["human_ground_truth"],
                "manufacturing_accuracy_validated": False,
            }
        )
    rows.sort(
        key=lambda row: (
            row["mean_silhouette_iou"] is not None,
            row["mean_silhouette_iou"] or -1.0,
            row["mean_external_boundary_f1_2px"] or -1.0,
        ),
        reverse=True,
    )
    return {"schema_version": SCHEMA_VERSION, "report_count": len(rows), "ranking": rows}


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = report["ranking"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["model_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="create a hardware-aware execution plan")
    plan_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    plan_parser.add_argument("--output", type=Path, default=Path("runs/benchmarks/2d_to_3d/plan.json"))

    validate_parser = subparsers.add_parser("validate", help="validate one imported model output")
    validate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)

    rank_parser = subparsers.add_parser("rank", help="rank comparable validation reports")
    rank_parser.add_argument("reports", nargs="+", type=Path)
    rank_parser.add_argument("--output", type=Path, required=True)
    rank_parser.add_argument("--csv", type=Path)

    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(load_registry(args.registry), detect_hardware(args.output.parent))
        _write_json(args.output, result)
    elif args.command == "validate":
        result = validate_manifest(args.manifest, args.registry)
        _write_json(args.output, result)
    else:
        result = rank_reports(args.reports)
        _write_json(args.output, result)
        if args.csv:
            _write_csv(args.csv, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
