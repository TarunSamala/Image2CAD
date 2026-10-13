"""Pinned VGGT adapter for Phase 3/B2 camera and dense-geometry evidence.

The adapter never downloads a checkpoint implicitly.  The caller must provide
a local checkpoint and declare its licence class so legal and runtime blockers
are retained in the benchmark record instead of being hidden.
"""

from __future__ import annotations

import json
import gc
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np


PINNED_VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
COMMERCIAL_CHECKPOINT_LICENSE = "VGGT-1B-Commercial"
NONCOMMERCIAL_CHECKPOINT_LICENSE = "VGGT-1B-NonCommercial"


def probe_vggt(
    checkpoint: str | Path | None,
    checkpoint_license_id: str | None,
    allow_noncommercial_research: bool = False,
    allow_low_vram_attempt: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint_path is None:
        blockers.append("local_checkpoint_not_supplied")
    elif not checkpoint_path.exists():
        blockers.append("local_checkpoint_not_found")

    permitted = checkpoint_license_id == COMMERCIAL_CHECKPOINT_LICENSE
    if (
        allow_noncommercial_research
        and checkpoint_license_id == NONCOMMERCIAL_CHECKPOINT_LICENSE
    ):
        permitted = True
    if not permitted:
        blockers.append("checkpoint_licence_not_approved_for_requested_use")

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info() if cuda_available else (0, 0)
        except RuntimeError:
            free_bytes, total_bytes = 0, 0
    except ImportError:
        torch_version = None
        cuda_available = False
        gpu_name = None
        free_bytes, total_bytes = 0, 0
        blockers.append("torch_not_installed")

    try:
        import vggt  # noqa: F401

        code_available = True
    except ImportError:
        code_available = False
        blockers.append("pinned_vggt_code_not_installed")

    if not cuda_available:
        blockers.append("cuda_not_available")
    elif total_bytes and total_bytes < 8 * 1024**3:
        if allow_low_vram_attempt:
            warnings.append("gpu_memory_below_conservative_8_gib_probe_gate")
        else:
            blockers.append("gpu_memory_below_conservative_8_gib_probe_gate")

    return {
        "backend": "vggt",
        "status": "ready" if not blockers else "blocked",
        "authority": "machine_hypothesis",
        "metric": False,
        "cross_view_validated": False,
        "pinned_code_commit": PINNED_VGGT_COMMIT,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_license_id": checkpoint_license_id,
        "noncommercial_research_override": bool(allow_noncommercial_research),
        "low_vram_attempt": bool(allow_low_vram_attempt),
        "code_available": code_available,
        "runtime": {
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_free_bytes": int(free_bytes),
            "gpu_total_bytes": int(total_bytes),
        },
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
    }


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_vggt_object(
    image_paths: list[Path],
    output_dir: Path,
    checkpoint: str | Path,
    checkpoint_license_id: str,
    device: str = "cuda",
    save_dense: bool = False,
    allow_noncommercial_research: bool = False,
    allow_low_vram_attempt: bool = False,
    camera_only: bool = False,
    image_size: int = 518,
) -> dict[str, Any]:
    probe = probe_vggt(
        checkpoint,
        checkpoint_license_id,
        allow_noncommercial_research,
        allow_low_vram_attempt,
    )
    if probe["status"] != "ready":
        return probe

    if image_size < 56 or image_size % 14:
        raise ValueError("VGGT image_size must be at least 56 and divisible by 14")
    if camera_only and save_dense:
        raise ValueError("Dense evidence is unavailable in camera-only mode")

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if device == "cuda" and not torch.cuda.is_available():
        return {**probe, "status": "blocked", "blockers": ["cuda_not_available"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    checkpoint_path = Path(checkpoint)
    checkpoint_sha256 = None if checkpoint_path.is_dir() else _sha256(checkpoint_path)
    dtype = torch.float16 if device == "cuda" and allow_low_vram_attempt else torch.float32
    model = None
    images = None
    try:
        torch.manual_seed(0)
        if device == "cuda":
            torch.cuda.manual_seed_all(0)
        if checkpoint_path.is_dir():
            if camera_only:
                raise ValueError("Camera-only low-VRAM mode requires a local checkpoint file")
            model = VGGT.from_pretrained(str(checkpoint_path), local_files_only=True)
        else:
            model = VGGT(
                enable_point=not camera_only,
                enable_depth=not camera_only,
                enable_track=not camera_only,
            )
            if dtype == torch.float16:
                model = model.half()
            load_options: dict[str, Any] = {
                "map_location": "cpu",
                "weights_only": True,
            }
            if allow_low_vram_attempt:
                load_options["mmap"] = True
            state = torch.load(checkpoint_path, **load_options)
            incompatible = model.load_state_dict(state, strict=not camera_only)
            del state
            gc.collect()
            if camera_only and incompatible.missing_keys:
                raise RuntimeError(
                    f"Camera-only checkpoint is missing required keys: {incompatible.missing_keys[:5]}"
                )
        model = model.to(device).eval()
        images, _ = load_and_preprocess_images_square(
            [str(path) for path in image_paths], target_size=image_size
        )
        images = images.to(device=device, dtype=dtype)
        autocast_enabled = device == "cuda"
        autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
        with torch.inference_mode(), torch.autocast(
            device_type=device,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            batched_images = images.unsqueeze(0)
            aggregated_tokens, patch_start_idx = model.aggregator(batched_images)
            pose_enc = model.camera_head(aggregated_tokens)[-1]
            predictions = {"pose_enc": pose_enc}
            if not camera_only:
                depth, depth_conf = model.depth_head(
                    aggregated_tokens,
                    images=batched_images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=1,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                world_points, world_points_conf = model.point_head(
                    aggregated_tokens,
                    images=batched_images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=1,
                )
                predictions["world_points"] = world_points
                predictions["world_points_conf"] = world_points_conf
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), images.shape[-2:]
        )
    except torch.cuda.OutOfMemoryError as exc:
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        model = None
        images = None
        gc.collect()
        torch.cuda.empty_cache()
        report = {
            **probe,
            "status": "failed_out_of_memory",
            "image_count": len(image_paths),
            "image_names": [path.name for path in image_paths],
            "camera_only": bool(camera_only),
            "image_size": image_size,
            "dtype": str(dtype),
            "checkpoint_sha256": checkpoint_sha256,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "peak_cuda_bytes": peak_cuda_bytes,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "warning": "Low-VRAM research attempt failed safely; no camera or geometry accuracy is claimed.",
        }
        (output_dir / "vggt_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    cameras_path = output_dir / "vggt_cameras.npz"
    np.savez_compressed(
        cameras_path,
        extrinsic=_numpy(extrinsic),
        intrinsic=_numpy(intrinsic),
    )
    dense_path = None
    dense_shapes = {}
    if save_dense:
        dense = {
            key: _numpy(value)
            for key, value in predictions.items()
            if key in {"depth", "depth_conf", "world_points", "world_points_conf"}
        }
        dense_shapes = {key: list(value.shape) for key, value in dense.items()}
        dense_path = output_dir / "vggt_dense_evidence.npz"
        np.savez_compressed(dense_path, **dense)

    report = {
        **probe,
        "status": "completed_machine_hypothesis",
        "image_count": len(image_paths),
        "image_names": [path.name for path in image_paths],
        "camera_shapes": {
            "extrinsic": list(_numpy(extrinsic).shape),
            "intrinsic": list(_numpy(intrinsic).shape),
        },
        "dense_shapes": dense_shapes,
        "camera_only": bool(camera_only),
        "image_size": image_size,
        "dtype": str(dtype),
        "checkpoint_sha256": checkpoint_sha256,
        "artifacts": {
            "cameras": str(cameras_path),
            "dense": str(dense_path) if dense_path else None,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        "warning": "VGGT outputs are uncalibrated machine hypotheses until compared with reviewed camera and geometry truth.",
    }
    (output_dir / "vggt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del predictions, images, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return report
