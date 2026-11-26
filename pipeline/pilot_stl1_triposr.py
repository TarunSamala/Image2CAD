"""Trace a pinned TripoSR shape-only run on the existing STL-1 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage.measure import marching_cubes

from pipeline.pilot_stl1_hunyuan import OBJECTS, OUTPUT as INPUTS, evaluate
from pipeline.run_saraswati_blind_benchmark import _sha256, _write_json


OUTPUT = Path("dataset/phase_runs/v1/stl1_triposr_pilot_v1")
MODEL_DIR = Path("models/triposr/5b52193")
CHECKPOINT_SHA = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
SOURCE_REV = "107cefdc244c39106fa830359024f6a2f1c78871"
WEIGHTS_REV = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
DENSITY_THRESHOLD = 25.0


def _cpu_surface(level, threshold):
    """Use CPU marching cubes in the coordinate order expected by TripoSR."""
    import torch

    vertices, faces, _, _ = marching_cubes(level.detach().float().cpu().numpy(), level=threshold)
    return torch.from_numpy(vertices[:, ::-1].copy()), torch.from_numpy(faces.copy())


def _runtime(device):
    import torch
    from omegaconf import OmegaConf

    # Keep the upstream density field and threshold. The CPU adapter replaces
    # only the compiled mesh extractor within this offline worker.
    mesher = ModuleType("torchmcubes")
    mesher.marching_cubes = _cpu_surface
    prior_mesher = sys.modules.get("torchmcubes")
    sys.modules["torchmcubes"] = mesher
    try:
        from tsr.models.tokenizers import image as tokenizer
        from tsr.system import TSR
    finally:
        if prior_mesher is None:
            sys.modules.pop("torchmcubes", None)
        else:
            sys.modules["torchmcubes"] = prior_mesher
    config = OmegaConf.load(MODEL_DIR / "config.yaml")
    OmegaConf.resolve(config)
    with patch.object(tokenizer, "hf_hub_download", return_value=str(MODEL_DIR / "dino-vitb16-config.json")):
        model = TSR(config)
    checkpoint = MODEL_DIR / "model.ckpt"
    if _sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("TripoSR checkpoint hash mismatch")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    model.renderer.set_chunk_size(8192)
    return model.eval().to(device)


def generate(object_ids, device, resolution):
    import torch
    from tsr.utils import resize_foreground

    print("Loading TripoSR weights", flush=True)
    model = _runtime(device)
    trace = {}

    def capture(name):
        def record(_module, _args, output):
            trace[name] = list(output.shape)
        return record

    model.image_tokenizer.register_forward_hook(capture("image_tokens_shape"))
    model.post_processor.register_forward_hook(capture("triplane_shape"))
    for object_id in object_ids:
        path = INPUTS / object_id / "input" / "front.png"
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        input_hash = _sha256(path)
        if manifest["inputs"]["front"]["sha256"] != input_hash:
            raise ValueError(f"Prepared front image changed: {path}")
        with Image.open(path) as image:
            rgba = np.asarray(resize_foreground(image.convert("RGBA"), 0.85), dtype=np.float32) / 255
        composited = rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])
        trace.clear()
        start = time.monotonic()
        with torch.inference_mode():
            scene_codes = model(composited, device)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_seconds = time.monotonic() - start
            surface = model.extract_mesh(scene_codes, has_vertex_color=False,
                                         resolution=resolution, threshold=DENSITY_THRESHOLD)[0]
        if not len(surface.faces) or not np.isfinite(surface.vertices).all():
            raise ValueError(f"No valid triangle surface for {object_id}")
        if surface.volume < 0:
            surface.invert()
        output = OUTPUT / object_id
        output.mkdir(parents=True, exist_ok=True)
        mesh_path = output / "triposr_shape.obj"
        surface.export(mesh_path)
        _write_json(output / "generation.json", {
            "model": "TripoSR", "model_revision": WEIGHTS_REV, "source_revision": SOURCE_REV,
            "input_views": ["front"], "front_sha256": input_hash, "device": device,
            "checkpoint_sha256": CHECKPOINT_SHA, "config_sha256": _sha256(MODEL_DIR / "config.yaml"),
            "input_preprocessing": "upstream foreground crop to 85% and gray compositing",
            "mesher": "scikit-image CPU marching cubes on upstream density samples",
            "density_threshold": DENSITY_THRESHOLD, "resolution": resolution,
            "mesh_sha256": _sha256(mesh_path), "trace": trace,
            "inference_seconds": round(inference_seconds, 3),
            "inference_and_extraction_seconds": round(time.monotonic() - start, 3),
        })
        print(f"Generated {object_id}: {len(surface.faces):,} triangles", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "evaluate"))
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resolution", type=int, default=192)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.objects, args.device, args.resolution)
    else:
        evaluate(args.objects, candidate_root=OUTPUT, generator="triposr")


if __name__ == "__main__":
    main()
