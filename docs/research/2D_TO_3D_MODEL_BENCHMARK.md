# 2D-to-3D model benchmark

This benchmark compares external image-to-3D generators without allowing any model's own preview or confidence score to decide whether its output is accurate.

## Scope

The initial registry contains these relevant model families:

- the existing five-view Image2CAD visual-hull baseline;
- Shap-E;
- TripoSR;
- Stable Fast 3D;
- SPAR3D;
- CRM;
- Wonder3D;
- Unique3D;
- InstantMesh;
- LGM;
- DreamGaussian;
- TRELLIS and TRELLIS.2;
- Hunyuan3D-2mv and Hunyuan3D 2.1;
- PartCrafter.

“All models” means all relevant, reproducible models admitted to the versioned registry. It cannot mean every repository using the phrase image-to-3D. A candidate is excluded until its official source, checkpoint, input contract and license can be identified.

The registry is [models.json](../../benchmarks/2d_to_3d/models.json). It contains no executable shell commands. This prevents a downloaded or edited registry from becoming an arbitrary command-execution mechanism.

## Why the models are isolated

These projects require incompatible Python, PyTorch, CUDA, xFormers and compiled-extension versions. Installing all of them into the current Image2CAD environment would make results irreproducible and would likely break the validated pipeline.

Each generator will therefore use one of:

- a pinned Docker image;
- a pinned container built from its official repository;
- a remote GPU job using the same pinned environment.

Model caches belong under `/home`, not the small root filesystem. Every adapter must record its source commit, checkpoint hash, code license and checkpoint license.

## Current laptop boundary

The detected GPU is an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM. At the time of planning, only about 2.7 GB was free. This is enough for the current visual-hull baseline but below the official practical requirements of most modern image-to-3D generators:

- TripoSR and Stable Fast 3D: approximately 6 GB;
- SPAR3D low-VRAM mode: approximately 7 GB;
- LGM: approximately 10 GB;
- TRELLIS: approximately 16 GB;
- TRELLIS.2: at least 24 GB.

CPU fallbacks can be recorded, but an extremely slow or memory-starved run is not a fair comparison. Large models should run on a remote 24 GB or larger GPU after the benchmark input and acceptance rules are frozen.

## STL-1 Hunyuan shape pilot

[The pilot module](../../pipeline/pilot_stl1_hunyuan.py) runs the pinned
[Hunyuan3D-2 source](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/tree/f8db63096c8282cb27354314d896feba5ba6ff8a)
and [2mv Turbo weights](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/3a761b539b29fe4ff64714813aa9560fd66f5de0).
Its local loader maps checkpoint tensors and offloads individual model
components. The shape-only 192-resolution configuration ran on the 4 GB
laptop GPU. The weights and runtime are cached under `models/hunyuan3d_2mv/`.
The pinned runtime adds Diffusers 0.32.2, Transformers 4.49.0, Accelerate
1.4.0, einops 0.8.1, safetensors 0.5.3 and PyMeshLab 2023.12.post3 to the
`image2cad-validation:phase0` image.

The module's `prepare`, `generate` and `evaluate` stages store inputs,
provenance, OBJs, renders and scores under
`dataset/phase_runs/v1/stl1_hunyuan_pilot_v1/`. Use
`generate --objects ring_001 ring_005 ring_004 --steps 8 --resolution 192`
to reproduce the pilot settings with the pinned source on `PYTHONPATH`.
Hunyuan uses front, left and right. The hull also uses top. The unused
angled view and paired OBJ are opened during evaluation. The aligned gray
renders, opening coverage and per-region silhouette scores help separate
visible relief from shape and connectivity errors. The [Tencent model
license](https://huggingface.co/tencent/Hunyuan3D-2mv/raw/main/LICENSE) applies.
The public checkout carries the benchmark code. Hunyuan outputs, transferred
relief meshes and comparison sheets containing either remain local.

### How Hunyuan uses the input views

The pinned preprocessor crops each image to its alpha foreground, rescales it
independently, and sorts views by their assigned camera slots. The slots are
front, left, back and right. The DINOv2 conditioner adds a positional embedding
for each slot to the corresponding image features. The diffusion transformer
generates the shape latent from those features. The VAE decodes a scalar field,
and marching cubes extracts its surface. This generation path receives the
view slots instead of measured camera intrinsics or poses.

The view ablation uses the same front image, checkpoint, seed, step count and
mesh resolution across both variants. It adds a trace of the actual
conditioning and diffusion tensor sizes to the generation record. Reproduce
the front-only runs with `generate --variant front-only --objects ring_001
ring_005 ring_004 --steps 8 --resolution 192`, then `evaluate` with the same
variant and ring IDs. Run `python -m pipeline.report_stl1_hunyuan_view_ablation`
to compare them with the original three-view runs.

### TripoSR shape and structure pilot

[The TripoSR adapter](../../pipeline/pilot_stl1_triposr.py) uses the same
prepared front view on Rings 001, 005 and 004. Its trace records 1025 image
tokens and three learned feature planes. The shape decoder queries density
in chunks before CPU marching cubes. The local runtime uses the published
checkpoint and threshold, with scikit-image replacing the compiled mesh
extractor. Use the module's `generate` and `evaluate` stages inside the
validation image with the pinned TripoSR source on `PYTHONPATH`.

[The source-image structure screen](../../pipeline/screen_stl1_generated_rings.py)
checks the projected top outline and the main enclosed opening in a common
hull frame, alongside mesh connectivity, winding and watertightness. It
keeps the unused angled comparison available separately. The thresholds
are provisional for the selected sculptural ring pilot.

## Standard experiment contract

An external model writes its output to a versioned directory:

```text
runs/benchmarks/2d_to_3d/<asset>/<model>/<run_id>/
├── input/
├── raw_output/
├── mesh.obj|glb|ply|stl
├── renders/
├── previews/
├── inspection.json
├── manifest.json
└── validation.json
```

The manifest passed to the independent validator uses this structure:

```json
{
  "model_id": "triposr",
  "input_mode": "single_image",
  "mesh_path": "mesh.obj",
  "source_images": {
    "front": "input/front.png"
  },
  "preview_review": {
    "status": "approved",
    "preview_path": "previews/six_view.png",
    "reviewed_by": "human",
    "notes": "The surface render is complete enough to continue."
  },
  "target_height_mm": 431.8,
  "metric_calibrated": true,
  "ground_truth_kind": "human_reviewed",
  "target_masks": {
    "front": "targets/front.png"
  },
  "rendered_masks": {
    "front": "renders/front.png"
  }
}
```

Accepted `ground_truth_kind` values should identify the real evidence source, for example `human_reviewed`, `machine_pseudo_mask` or `none`. Only `human_reviewed` targets can validate visible-view accuracy.

## Preview gate

Every generated OBJ or GLB must be rendered independently from at least six fixed views before numerical validation. The preview must be a depth-buffered surface render of the actual exported file, not a sparse point sample and not the generator website's screenshot.

The reviewer records `approved` or `rejected` in `preview_review`. A rejected model stops here and is retained as failed research evidence. The command-line validator rejects missing, pending or rejected preview reviews by default. `--allow-unreviewed-preview` exists only for explicitly labelled legacy diagnostics.

The preview gate checks for catastrophic failures that metrics should never legitimise: duplicated or stretched limbs, torn surfaces, collapsed depth, severe holes, texture projection errors, missing primary components and obvious cross-view fusion.

## Common validation

Every imported mesh is checked for:

- readable triangle geometry;
- finite vertices;
- vertex and face count;
- connected body count;
- degenerate faces;
- consistent winding;
- watertightness;
- bounds, extents and volume where meaningful;
- file hash and byte size;
- scale factor required to reach the requested physical height.

When comparable target and rendered masks are available, the validator reports per-view:

- silhouette IoU;
- external-boundary F1 at two pixels;
- mean IoU;
- lowest-view IoU;
- mean boundary F1.

The numerical visual gate is the current Phase 3 minimum: mean IoU at least `0.80`, every supplied view at least `0.70`, and mean boundary F1 at least `0.75`. Passing those numbers against pseudo-masks does not validate accuracy.

No imported generative mesh is labelled manufacturing accurate. Hidden surfaces are inferred, physical scale may contain only one known dimension, and OBJ/GLB mesh topology is not equivalent to editable exact CAD.

## Commands

Create a hardware-aware plan:

```bash
PYTHONPATH=pipeline python pipeline/benchmark_2d_to_3d.py plan \
  --output runs/benchmarks/2d_to_3d/plan.json
```

Validate one imported model output:

```bash
PYTHONPATH=pipeline python pipeline/benchmark_2d_to_3d.py validate \
  --manifest runs/benchmarks/2d_to_3d/elephant/triposr/run-001/manifest.json \
  --output runs/benchmarks/2d_to_3d/elephant/triposr/run-001/validation.json
```

Rank comparable reports:

```bash
PYTHONPATH=pipeline python pipeline/benchmark_2d_to_3d.py rank \
  runs/benchmarks/2d_to_3d/elephant/*/*/validation.json \
  --output runs/benchmarks/2d_to_3d/elephant/ranking.json \
  --csv runs/benchmarks/2d_to_3d/elephant/ranking.csv
```

Only runs using the same source images, reviewed masks, camera-registration method and scoring version belong in one ranking.

## Execution order

1. Save the elephant reference as original full-resolution files rather than a screenshot embedded in chat.
2. Split and review the six views; record that the source views contain inconsistent ornament details.
3. Create human-reviewed silhouettes for visible-view scoring.
4. Run the existing visual hull as the deterministic multi-view baseline.
5. Run Shap-E as an older open baseline, initially on CPU only if a bounded smoke test is practical.
6. Run TripoSR, Stable Fast 3D and SPAR3D on a remote GPU.
7. Run CRM, Wonder3D, Unique3D and InstantMesh in separate pinned environments.
8. Run LGM, DreamGaussian, TRELLIS, TRELLIS.2 and Hunyuan variants remotely.
9. Run PartCrafter to evaluate whether generated part separation helps component recovery.
10. Independently render the actual exported model and record the preview decision.
11. Validate and rank only outputs whose preview review is approved.
12. Use the strongest mesh only as Phase 3 evidence; continue with reviewed components, exact scale and CAD reconstruction.

## Decision rule

A new model is useful only if it provides a measurable improvement over the current baseline without introducing unacceptable topology, licensing, runtime or memory costs. Several model outputs may be fused if their evidence is complementary, but scores must also be reported separately so a weak model cannot hide inside an ensemble.
