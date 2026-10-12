"""Run one metric-depth backend into the Phase 2 consensus contract.

The module deliberately imports backend-specific libraries lazily.  Each model
runs in its own pinned container and writes the same small, auditable contract:
one float32 depth map per input image plus a backend report.  A value expressed
in metres is still only a model hypothesis until physical scale is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


BACKEND_MODELS: dict[str, dict[str, Any]] = {
    "depth_anything_v2_metric": {
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        "revision": "3bc65d4e14a6786a61acec16453c50e12bf5f338",
        "license": "Apache-2.0",
        "checkpoint_role": "indoor metric-depth hypothesis",
    },
    "unidepth_v2": {
        "model_id": "lpiccinelli/unidepth-v2-vits14",
        "revision": "038c238f06c87b6c2f5b3749fd51fbf442b1f218",
        "license": "CC-BY-NC-4.0",
        "checkpoint_role": "research-only metric-depth and camera hypothesis",
    },
    "metric3d_v2": {
        "model_id": "JUGGHM/Metric3D metric_depth_vit_small_800k",
        "revision": "Metric3D code eb5b6fac0dc155e4e52f576e304fbf11655ff339",
        "license": "BSD-2-Clause code; checkpoint terms require separate review",
        "checkpoint_role": "metric-depth and surface-normal hypothesis",
    },
    "depth_pro": {
        "model_id": "apple/DepthPro-hf",
        "revision": "de816c8ce7168afcb231f96d501d72b869d0beda",
        "license": "Apple-ASCL",
        "checkpoint_role": "metric-depth and focal-length hypothesis",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_evidence(path: Path) -> dict[str, Any]:
    """Return compact content evidence for a checkpoint file or local model folder."""
    if path.is_file():
        return {
            "kind": "file",
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    aggregate = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest = sha256(item)
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": item.stat().st_size,
            }
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if not files:
        raise ValueError(f"Checkpoint directory is empty: {path}")
    return {
        "kind": "directory",
        "sha256": aggregate.hexdigest(),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).squeeze()
    while result.ndim > 2:
        result = result[0]
    return result


def _resize(values: np.ndarray, height: int, width: int) -> np.ndarray:
    if values.shape != (height, width):
        values = cv2.resize(values, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.asarray(values, dtype=np.float32)


def load_depth_anything_v2(checkpoint: Path, device: str) -> Callable[[Path], dict[str, Any]]:
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    config = {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "max_depth": 20,
    }
    model = DepthAnythingV2(**config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    def infer(path: Path) -> dict[str, Any]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        with torch.inference_mode():
            depth = model.infer_image(image, input_size=518)
        return {"depth_m": np.asarray(depth, dtype=np.float32)}

    return infer


def load_unidepth_v2(checkpoint: Path, device: str) -> Callable[[Path], dict[str, Any]]:
    import torch
    from PIL import Image
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(str(checkpoint), local_files_only=True)
    model = model.to(device).eval()

    def infer(path: Path) -> dict[str, Any]:
        rgb = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)
        with torch.inference_mode():
            prediction = model.infer(rgb.to(device))
        result: dict[str, Any] = {"depth_m": _to_numpy(prediction["depth"])}
        if prediction.get("confidence") is not None:
            result["confidence"] = _to_numpy(prediction["confidence"])
        if prediction.get("intrinsics") is not None:
            result["intrinsics"] = _to_numpy(prediction["intrinsics"])
        return result

    return infer


def load_metric3d_v2(
    checkpoint: Path,
    device: str,
    assumed_horizontal_fov_degrees: float,
) -> Callable[[Path], dict[str, Any]]:
    import torch

    repo = os.environ.get("METRIC3D_CODE_DIR", "/opt/metric3d")
    if not (Path(repo) / "hubconf.py").is_file():
        raise FileNotFoundError(f"Metric3D code checkout is unavailable at {repo}")
    model = torch.hub.load(
        repo,
        "metric3d_vit_small",
        source="local",
        pretrain=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model = model.to(device).eval()
    input_height, input_width = 616, 1064
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    def infer(path: Path) -> dict[str, Any]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot decode {path}")
        rgb = bgr[:, :, ::-1]
        height, width = rgb.shape[:2]
        scale = min(input_height / height, input_width / width)
        resized = cv2.resize(
            rgb,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_h, pad_w = input_height - resized.shape[0], input_width - resized.shape[1]
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53],
        )
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float()
        tensor = ((tensor - mean) / std)[None].to(device)
        with torch.inference_mode():
            pred_depth, confidence, output = model.inference({"input": tensor})
        pred_depth = pred_depth.squeeze()[top : input_height - bottom, left : input_width - right]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None], (height, width), mode="bilinear", align_corners=False
        ).squeeze()
        focal_px = 0.5 * width / math.tan(math.radians(assumed_horizontal_fov_degrees) / 2.0)
        focal_scaled = focal_px * scale
        depth_m = pred_depth * (focal_scaled / 1000.0)
        result: dict[str, Any] = {
            "depth_m": _to_numpy(torch.clamp(depth_m, 0, 300)),
            "assumed_focal_length_px": float(focal_px),
        }
        if confidence is not None:
            confidence = confidence.squeeze()[top : input_height - bottom, left : input_width - right]
            result["confidence"] = _to_numpy(
                torch.nn.functional.interpolate(
                    confidence[None, None], (height, width), mode="bilinear", align_corners=False
                ).squeeze()
            )
        normal = output.get("prediction_normal")
        if normal is not None:
            normal = normal[0, :3, top : input_height - bottom, left : input_width - right]
            normal = torch.nn.functional.interpolate(
                normal[None], (height, width), mode="bilinear", align_corners=False
            )[0]
            result["normals"] = normal.float().cpu().numpy().transpose(1, 2, 0)
        return result

    return infer


def load_depth_pro(checkpoint: Path, device: str) -> Callable[[Path], dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        str(checkpoint), local_files_only=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    def infer(path: Path) -> dict[str, Any]:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        processed = processor.post_process_depth_estimation(
            output, target_sizes=[(image.height, image.width)]
        )[0]
        result: dict[str, Any] = {"depth_m": _to_numpy(processed["predicted_depth"])}
        for key in ("focal_length", "field_of_view"):
            if processed.get(key) is not None:
                value = processed[key]
                if hasattr(value, "detach"):
                    value = value.detach().float().cpu().item()
                result[key] = float(value)
        return result

    return infer


LOADERS = {
    "depth_anything_v2_metric": load_depth_anything_v2,
    "unidepth_v2": load_unidepth_v2,
    "depth_pro": load_depth_pro,
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = (root / args.output_dir / args.backend).resolve()
    report_path = output / "backend_report.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"{report_path} exists; pass --force to replace this backend run")
    output.mkdir(parents=True, exist_ok=True)

    if args.backend == "metric3d_v2":
        infer = load_metric3d_v2(
            args.checkpoint.resolve(), args.device, args.assumed_horizontal_fov_degrees
        )
    else:
        infer = LOADERS[args.backend](args.checkpoint.resolve(), args.device)

    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except ImportError as error:
        raise RuntimeError("PyTorch is required by metric-depth backends") from error

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    selected_samples = manifest["samples"]
    if args.sample_keys:
        requested = set(args.sample_keys)
        selected_samples = [sample for sample in selected_samples if sample["sample_key"] in requested]
        missing = requested - {sample["sample_key"] for sample in selected_samples}
        if missing:
            raise ValueError(f"Unknown sample keys: {sorted(missing)}")
    if args.max_samples is not None:
        selected_samples = selected_samples[: args.max_samples]
    for sample in selected_samples:
        image_path = root / sample["image_path"]
        mask = cv2.imread(str(root / sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot decode mask for {sample['sample_key']}")
        view_started = time.perf_counter()
        prediction = infer(image_path)
        depth = _resize(_to_numpy(prediction.pop("depth_m")), *mask.shape)
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Invalid non-positive or non-finite depth for {sample['sample_key']}")
        sample_dir = output / sample["object_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        depth_path = sample_dir / f"{sample['view']}_depth_m.npy"
        np.save(depth_path, depth.astype(np.float32))
        artifacts: dict[str, str] = {
            "depth_m": str(depth_path.relative_to(root)),
        }
        for key in ("confidence", "intrinsics", "normals"):
            value = prediction.pop(key, None)
            if value is None:
                continue
            value_path = sample_dir / f"{sample['view']}_{key}.npy"
            np.save(value_path, np.asarray(value, dtype=np.float32))
            artifacts[key] = str(value_path.relative_to(root))
        masked = depth[mask > 127]
        records.append(
            {
                "sample_key": sample["sample_key"],
                "object_id": sample["object_id"],
                "view": sample["view"],
                "source_sha256": sample["source_sha256"],
                "artifacts": artifacts,
                "depth_sha256": sha256(depth_path),
                "depth_shape": list(depth.shape),
                "depth_unit": "metre_model_hypothesis",
                "masked_depth_m": {
                    "p02": float(np.percentile(masked, 2)),
                    "median": float(np.median(masked)),
                    "p98": float(np.percentile(masked, 98)),
                },
                "runtime_seconds": round(time.perf_counter() - view_started, 6),
                "auxiliary": prediction,
            }
        )

    report = {
        "schema_version": "image2cad_metric_depth_backend_v1",
        "created_at_utc": utc_now(),
        "backend": args.backend,
        "model": BACKEND_MODELS[args.backend],
        "device": args.device,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_evidence": checkpoint_evidence(args.checkpoint.resolve()),
        "input_manifest": str(manifest_path.relative_to(root)),
        "input_manifest_sha256": sha256(manifest_path),
        "sample_count": len(records),
        "requested_sample_count": manifest["sample_count"],
        "coverage_complete": len(records) == manifest["sample_count"],
        "records": records,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
        "claims": {
            "model_outputs_metric_units": True,
            "metric_accuracy_validated_on_jewellery": False,
            "physical_scale_validated": False,
            "manufacturing_measurements_validated": False,
        },
        "decision": "research_only_model_hypothesis",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1/input_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1/backends"),
    )
    parser.add_argument("--backend", choices=tuple(BACKEND_MODELS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--assumed-horizontal-fov-degrees", type=float, default=55.0)
    parser.add_argument("--sample-keys", nargs="*")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
