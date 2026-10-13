# Phase 3/B2 camera and geometry evidence

This checkpoint follows the project Bible's Phase 3 order. It adds two independent
camera/geometry backends without promoting their output to CAD ground truth:

- **COLMAP / PyCOLMAP 4.2.1**: SIFT features, exhaustive matching, geometric
  verification, incremental camera recovery and sparse points. Exhaustive matching
  is appropriate here because each STL-1 object has only five views.
- **VGGT** pinned to commit `a288dd0f14786c93483e45524328726ab7b1b4ce`:
  camera, depth and point-map hypotheses through a local-checkpoint-only adapter.
  No model is downloaded automatically. Commercial use requires the separately
  gated commercial checkpoint and its terms to be reviewed.

The current public-deployment and hardware audit is recorded in
[Cross-view backend readiness](CROSS_VIEW_READINESS.md). DUSt3R, MASt3R and
VGGT-Omega are retained as research references rather than public-product
dependencies because their official licences are non-commercial. VGGT remains
the executable path after the operator obtains the gated commercial checkpoint
and compatible hardware.

A separate bounded research experiment also proves that the original
noncommercial VGGT checkpoint can execute five Ring01 views at 224 px on this
4 GiB GPU. It generated cameras, depth and world points with a 2.54 GiB peak,
but the fused point cloud was rejected as reconstruction geometry because of
cross-view ghosting and floating outliers. See
`dataset/phase_runs/v1/ring01_vggt_dense_224_v1/` and do not confuse runtime
feasibility with geometry accuracy or product eligibility.

The same low-VRAM path has now been evaluated on three additional STL-1
objects: `ring_001` (dense face/relief, train), `ring_007` (complex ornament,
test) and `ring_023` (floral geometry, test). All 15 views completed below
4 GiB. Validation removes low-confidence points, keeps only geometry supported
by at least two views, then excludes the target view before reprojecting into
it. Mean leave-one-view-out silhouette IoU was `0.156287`, `0.158126` and
`0.105821`; all three were rejected at the `0.80` mean / `0.70` minimum gate.
The previews confirm camera/view disagreement rather than a small cloud-noise
problem.

An object-centric reframing refinement was then run on the same three objects
with identical model, resolution and validation settings. It improved
`ring_007` by `+0.002604` and `ring_023` by `+0.017053`, but reduced
`ring_001` by `-0.062892`. Cohort mean fell from `0.140078` to `0.125666`, so
the candidate was rejected and the full-frame result remains the selected
baseline. Both attempts and their transform manifests are retained; neither is
approved geometry.

External real-image intake remains part of Phase 1/2 generalisation testing:
206 OrnAsia and Segment Your Ring images have been processed. They are not fed
to the multi-view geometry backend because the sources are single-view or lack
verified same-object/view identity. Combining unrelated photographs would
create invalid 3D evidence.

## Reproducible runs

Build COLMAP:

```bash
docker build -f docker/geometry-benchmark/Dockerfile.colmap \
  -t image2cad-geometry-colmap:4.2.1 .
```

Build the combined pinned VGGT/PyCOLMAP runtime:

```bash
docker build -f docker/geometry-benchmark/Dockerfile.vggt \
  -t image2cad-geometry-vggt:a288dd0 .
```

Run all STL-1 COLMAP experiments and record a blocked VGGT probe:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-geometry-vggt:a288dd0 \
  python -m pipeline.run_phase3_geometry_benchmark --backend all --force
```

A legal local VGGT checkpoint can be evaluated in the pinned VGGT container with
`--vggt-checkpoint` and `--vggt-license-id VGGT-1B-Commercial`. The original
non-commercial checkpoint is rejected unless the run is explicitly marked as
non-commercial research.

The VGGT image also contains the same pinned PyCOLMAP wheel so `--backend all`
records both backend states from one GPU-visible environment.
It is a backend runtime, not the Phase 0 full-regression image; run the complete
application suite in `image2cad-geometry-colmap:4.2.1` and the geometry contract
tests in the VGGT image.

Run the representative multi-ring VGGT benchmark and independent validator:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-geometry-vggt:a288dd0 \
  python -m pipeline.run_vggt_multiring_experiment \
  --checkpoint /workspace/models/vggt/VGGT-1B/model.pt

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.validate_vggt_multiring_experiment
```

## Gate

STL-1 is still Tier A image evidence. It has no reviewed camera truth, known
physical dimension, paired CAD/scan truth or human-approved component graph.
Therefore COLMAP/VGGT outputs are machine hypotheses and the decision remains
`research_only`. After reviewed B2 evidence exists, the Bible's next phase is
Phase 4: the universal component and constraint graph.

## STL-1 baseline result

The retained 24-object run extracted `136,472` SIFT keypoints from all `120`
views. It produced `31` tentative descriptor matches but `0` geometrically
verified correspondences, so no COLMAP sparse camera model could be formed.
This is a useful failure result: the supplied views do not currently provide
the cross-view photometric correspondence required by classical structure from
motion. It does not mean the jewellery has no recoverable geometry; it means the
next experiment must use learned cross-view matching/VGGT and reviewed capture
evidence rather than treating COLMAP as successful.
