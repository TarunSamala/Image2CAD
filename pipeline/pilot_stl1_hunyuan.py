"""Prepare, generate, and compare a shape-only STL-1 Hunyuan3D-2mv pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from trimesh.sample import sample_surface

from pipeline.benchmark_stl1_four_view_holdout import (
    AUDIT_SIZE, MARGIN, OUTPUT as HULL_ROOT, _load_upright, _normalize_mask, _project, _score,
)
from pipeline.reconstruct_dataset_phase3 import _cell, _error_overlay, _manifest, _read
from pipeline.refine_phase3_1 import _jewelry_transform
from pipeline.run_saraswati_blind_benchmark import (
    _align, _foreground_mask, _load_mesh, _normalize, _sha256, _surface_metrics, _write_json,
)
from pipeline.train_dataset_phases import _binary_iou
from pipeline.validate_phase3_2 import _basis_camera, _clean_render


OUTPUT = Path("dataset/phase_runs/v1/stl1_hunyuan_pilot_v1")
OBJECTS = ("ring_001", "ring_005", "ring_004")
VIEWS = {"front": "front", "left": "lsv", "right": "rsv"}
MODEL_REVISION = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
SOURCE_REVISION = "f8db63096c8282cb27354314d896feba5ba6ff8a"
MODEL_DIR = Path("models/hunyuan3d_2mv/3a761b5/hunyuan3d-dit-v2-mv-turbo")
WEIGHTS_SHA256 = "172d7a989b99f66af760da0080d8c6f6350c72609536cf0c55b87a4afa574e45"


def prepare() -> None:
    records = {record["object_id"]: record for record in _manifest(Path("dataset/prepared_v1"))}
    for object_id in OBJECTS:
        item = records[object_id]
        folder = OUTPUT / object_id / "input"
        folder.mkdir(parents=True, exist_ok=True)
        inputs = {}
        for key, view in VIEWS.items():
            source = Path(item["views"][view]["image_path"])
            alpha = _foreground_mask(source)[0].astype(np.uint8) * 255
            rgba = cv2.cvtColor(_read(source), cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha
            image = folder / f"{key}.png"
            if not cv2.imwrite(str(image), rgba):
                raise OSError(f"Could not write {image}")
            inputs[key] = {"path": str(image), "sha256": _sha256(image),
                           "source_view": view, "source_sha256": _sha256(source)}
        _write_json(folder / "manifest.json", {
            "object_id": object_id, "split": item["split"], "inputs": inputs,
            "background": "same foreground extraction as the Studio hull",
        })
    print(f"Prepared {len(OBJECTS)} rings under {OUTPUT}", flush=True)


def _load_pipeline(config_path: Path, checkpoint: Path):
    """Assign mmap-backed checkpoint weights without allocating a float32 copy."""
    from accelerate import init_empty_weights
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config
    from safetensors import safe_open
    import torch
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    components = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        keys = weights.keys()
        for component in ("model", "vae", "conditioner"):
            prefix = f"{component}."
            with init_empty_weights(include_buffers=False):
                module = instantiate_from_config(config[component])
            state = {key[len(prefix):]: weights.get_tensor(key) for key in keys if key.startswith(prefix)}
            module.load_state_dict(state, strict=(component != "vae"), assign=True)
            missing = [name for name, parameter in module.named_parameters() if parameter.is_meta]
            if component == "vae" and missing and all(
                name.startswith(("encoder.", "pre_kl.")) for name in missing
            ):
                # The Turbo checkpoint ships the decoder. The unused mesh encoder
                # is only needed to convert an existing 3D surface to latents.
                module.encoder = None
                module.pre_kl = None
            elif missing:
                raise ValueError(f"{component} has missing parameters: {missing[:5]}")
            components[component] = module.eval()
    return Hunyuan3DDiTFlowMatchingPipeline(
        **components, scheduler=instantiate_from_config(config["scheduler"]),
        image_processor=instantiate_from_config(config["image_processor"]),
        device="cpu", dtype=torch.float16,
    )


def _variant_folder(object_id: str, variant: str) -> Path:
    root = OUTPUT / object_id
    return root if variant == "three-view" else root / variant


def generate(object_ids: list[str], device: str, steps: int, resolution: int,
             variant: str = "three-view") -> None:
    from accelerate import cpu_offload_with_hook
    import torch

    config = MODEL_DIR / "config.yaml"
    checkpoint = MODEL_DIR / "model.fp16.safetensors"
    if _sha256(checkpoint) != WEIGHTS_SHA256:
        raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
    print("Loading Hunyuan3D-2mv Turbo weights", flush=True)
    pipeline = _load_pipeline(config, checkpoint)
    hook = None
    if device == "cuda":
        # Keep one component on the GPU at a time. The upstream sampler calls
        # them in this order, and the VAE stays on-device for its mesh decoder.
        for name in ("conditioner", "model", "vae"):
            component, hook = cpu_offload_with_hook(getattr(pipeline, name), device, hook)
            setattr(pipeline, name, component)
        pipeline.device = torch.device("cuda")
    trace = {}

    def record_conditioner(_module, _args, kwargs, output):
        trace["encoder_input_shape"] = list(kwargs["image"].shape)
        trace["view_indices"] = [list(indices) for indices in kwargs["view_idxs"]]
        trace["condition_tensor_shapes"] = {
            name: list(tensor.shape) for name, tensor in output.items()
            if isinstance(tensor, torch.Tensor)
        }

    def record_latents(_step, _timestep, output):
        trace["diffusion_latent_shape"] = list(output.prev_sample.shape)

    pipeline.conditioner.register_forward_hook(record_conditioner, with_kwargs=True)
    input_keys = ("front",) if variant == "front-only" else tuple(VIEWS)
    for object_id in object_ids:
        input_folder = OUTPUT / object_id / "input"
        folder = _variant_folder(object_id, variant)
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = input_folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        images = {}
        for key in input_keys:
            path = input_folder / f"{key}.png"
            if _sha256(path) != manifest["inputs"][key]["sha256"]:
                raise ValueError(f"Input changed: {path}")
            with Image.open(path) as image:
                images[key] = image.convert("RGBA")
        start = time.monotonic()
        trace.clear()
        mesh = pipeline(
            image=images, num_inference_steps=steps, octree_resolution=resolution,
            num_chunks=2048, generator=torch.Generator(device=device).manual_seed(12345),
            output_type="trimesh", enable_pbar=False,
            callback=record_latents, callback_steps=1,
        )[0]
        if hook is not None:
            hook.offload()
        if mesh is None or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Hunyuan produced no valid triangle surface for {object_id}")
        output = folder / "hunyuan_shape.obj"
        mesh.export(output)
        _write_json(folder / "generation.json", {
            "model": "Hunyuan3D-2mv Turbo", "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION, "input_manifest_sha256": _sha256(manifest_path),
            "input_views": list(input_keys), "seed": 12345, "steps": steps, "octree_resolution": resolution,
            "device": device, "config_sha256": _sha256(config), "checkpoint_sha256": WEIGHTS_SHA256,
            "mesh_sha256": _sha256(output),
            "inference_seconds": round(time.monotonic() - start, 2),
            "conditioning_trace": trace,
        })
        print(f"Generated {object_id}: {len(mesh.faces):,} triangles", flush=True)


def _align_mesh(mesh, reference):
    """Use only the four-input hull to position the candidate for angled scoring."""
    samples, _ = sample_surface(mesh, 12000, seed=123)
    reference_samples, _ = sample_surface(reference, 12000, seed=456)
    source, center, scale = _normalize(samples)
    target, _, _ = _normalize(reference_samples)
    aligned, fit = _align(source, target)
    rotation = np.asarray(fit["rotation_source_to_target"])
    translation = (aligned - source @ rotation.T).mean(axis=0)
    mesh.vertices = ((mesh.vertices - center) / scale) @ rotation.T + translation
    return {**fit, "source_center": center.tolist(), "source_extent": scale,
            "normalized_translation": translation.tolist()}


def _proxy_metrics(mesh, proxy):
    points, _, _ = _normalize(sample_surface(mesh, 20000, seed=123)[0])
    proxy_points, _, _ = _normalize(sample_surface(proxy, 20000, seed=456)[0])
    aligned, _ = _align(points, proxy_points)
    return _surface_metrics(aligned, proxy_points)


def evaluate(object_ids: list[str], variant: str = "three-view", *,
             candidate_root: Path | None = None, generator: str = "hunyuan") -> None:
    hull = json.loads((HULL_ROOT / "report.json").read_text(encoding="utf-8"))
    rows = {row["object_id"]: row for row in hull["objects"]}
    archive = {row["object_id"]: row for row in _manifest(Path("dataset/prepared_v1"))}
    a, e = np.deg2rad([hull["camera"]["azimuth"], hull["camera"]["elevation"]])
    camera = _basis_camera("stl1_iso", (np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)))
    frame = (MARGIN, MARGIN, AUDIT_SIZE - MARGIN - 1, AUDIT_SIZE - MARGIN - 1)
    for object_id in object_ids:
        folder = _variant_folder(object_id, variant) if candidate_root is None else candidate_root / object_id
        row = rows[object_id]
        model_path = folder / f"{generator}_shape.obj"
        generation = json.loads((folder / "generation.json").read_text(encoding="utf-8"))
        if generation["mesh_sha256"] != _sha256(model_path):
            raise ValueError(f"Candidate mesh changed: {model_path}")
        reference = _load_upright(Path(row["mesh"]))
        candidate = _load_mesh(model_path)
        alignment = _align_mesh(candidate, reference)
        iso_path = Path(row["iso_path"])
        if row["iso_sha256"] != _sha256(iso_path):
            raise ValueError(f"Angled reference changed: {iso_path}")
        target = _normalize_mask(_foreground_mask(iso_path)[0])
        source_dir = Path("dataset/STL-1") / archive[object_id]["source_object_name"]
        proxies = list(source_dir.glob("*.obj"))
        if len(proxies) != 1:
            raise ValueError(f"Expected one paired OBJ in {source_dir}")
        proxy = _load_mesh(proxies[0])
        metrics = {}
        panels = []
        for name, mesh in (("visual_hull", reference), (generator, candidate)):
            projection = _project(mesh, camera)
            transform = _jewelry_transform(mesh, camera, frame)
            shaded = _clean_render(mesh, camera, target.shape, transform)
            shaded = cv2.cvtColor(shaded, cv2.COLOR_BGR2GRAY)
            panels.append(np.hstack([
                _cell(_read(iso_path), "UNUSED ANGLED IMAGE"),
                _cell(shaded, name.replace("_", " ").upper()),
                _cell(_error_overlay(target, projection), "SILHOUETTE OVERLAP"),
            ]))
            # In the pilot's angled references, the sculpted head faces left
            # and the shank extends right. These are image zones, not part masks.
            band_width = round(0.4 * (AUDIT_SIZE - 2 * MARGIN))
            head = slice(0, MARGIN + band_width)
            shank = slice(AUDIT_SIZE - MARGIN - band_width, AUDIT_SIZE)
            metrics[name] = {
                "angled": _score(target, projection), "faces": len(mesh.faces),
                "head_side_iou": round(_binary_iou(target[:, head], projection[:, head]), 6),
                "shank_side_iou": round(_binary_iou(target[:, shank], projection[:, shank]), 6),
                "watertight": bool(mesh.is_watertight), "connected_bodies": int(mesh.body_count),
                "paired_obj_agreement": _proxy_metrics(mesh, proxy),
            }
        preview = folder / "shape_comparison.png"
        if not cv2.imwrite(str(preview), np.vstack(panels)):
            raise OSError(f"Could not write {preview}")
        report = {
            "object_id": object_id, "split": row["split"],
            "input_view_count": {generator: len(generation["input_views"]), "visual_hull": 4},
            "evaluation_camera": {key: hull["camera"][key] for key in ("azimuth", "elevation")},
            "candidate_sha256": generation["mesh_sha256"], "angled_reference_sha256": row["iso_sha256"],
            f"{generator}_alignment": alignment, "alignment_reference": row["mesh"],
            "paired_obj": {"path": str(proxies[0]), "role": "secondary generated comparison mesh"},
            "metrics": metrics, "preview": str(preview),
        }
        _write_json(folder / "comparison.json", report)
        print(json.dumps({name: value["angled"]["silhouette_iou"] for name, value in metrics.items()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "evaluate"))
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=[OBJECTS[0]])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--variant", choices=("three-view", "front-only"), default="three-view")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "generate":
        generate(args.objects, args.device, args.steps, args.resolution, args.variant)
    else:
        evaluate(args.objects, args.variant)


if __name__ == "__main__":
    main()
