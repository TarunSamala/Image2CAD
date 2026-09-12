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

The registry is [models.json](../benchmarks/2d_to_3d/models.json). It contains no executable shell commands. This prevents a downloaded or edited registry from becoming an arbitrary command-execution mechanism.

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

## Standard experiment contract

An external model writes its output to a versioned directory:

```text
runs/benchmarks/2d_to_3d/<asset>/<model>/<run_id>/
├── input/
├── raw_output/
├── mesh.obj|glb|ply|stl
├── renders/
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
10. Independently render, validate and rank every successful output.
11. Use the strongest mesh only as Phase 3 evidence; continue with reviewed components, exact scale and CAD reconstruction.

## Decision rule

A new model is useful only if it provides a measurable improvement over the current baseline without introducing unacceptable topology, licensing, runtime or memory costs. Several model outputs may be fused if their evidence is complementary, but scores must also be reported separately so a weak model cannot hide inside an ensemble.
