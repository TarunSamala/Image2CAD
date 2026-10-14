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
That score is a surface-completion diagnostic: it combines camera alignment
with the inability of four visible-surface maps to reconstruct geometry hidden
from those views. It must not be reported as an isolated camera-pose score.

An object-centric reframing refinement was then run on the same three objects
with identical model, resolution and validation settings. It improved
`ring_007` by `+0.002604` and `ring_023` by `+0.017053`, but reduced
`ring_001` by `-0.062892`. Cohort mean fell from `0.140078` to `0.125666`, so
the candidate was rejected and the full-frame result remains the selected
baseline. Both attempts and their transform manifests are retained; neither is
approved geometry.

A semantic-view diagnostic compared VGGT optical-axis relationships with the
named front, top, isometric, left-side and right-side directions. Across the
three-ring cohort, mean pairwise angular error was `7.744514` degrees. A
leak-reduced named-axis silhouette holdout reached `0.705611` mean IoU, but its
minimum was only `0.273760`, so it was not accepted as calibrated camera
initialization. Directly snapping VGGT cameras to the nominal axes made the
surface-completion score worse (`0.140078` to `0.068415`) and was rejected.
The retained conclusion is narrower: VGGT's relative camera layout is broadly
plausible, while hidden-surface completion, reflective detail and several side
view relationships remain unresolved.

A resolution ladder was then evaluated at `224`, `280`, `336`, `448` and the
VGGT-native `518` px input size. Complete train/test cohort runs were retained
at `224`, `336`, `448` and `518`; the `280` px point is a training-ring probe.
The final 448-vs-518 selection uses `ring_004`, `ring_012` and `ring_020`, all
from the validation split. Validation mean IoU increased from `0.428126` to
`0.481936`, selecting 518 px. On the two test rings, mean surface-completion
IoU improved from `0.131973`
at 224 px to `0.491858` at 518 px. Mean projected-surface precision improved
from `0.467630` to `0.843920`, and completion recall from `0.155124` to
`0.541096`. Peak CUDA allocation was `3.122904` GiB, so 518 px is now the
selected low-VRAM research baseline. It is not approved geometry: all three
objects still fail the `0.80` mean / `0.70` minimum exit gate, and the weakest
held-out view remains `0.232783` IoU.

The test outputs were already observed during exploratory development before
the validation-only selection rule was formalized. They are therefore useful
evaluation evidence, but not a final unbiased generalization estimate; that
requires a new external identity-disjoint multi-view cohort.

The ladder also exposed retained CUDA tensors between sequential objects at
518 px. The adapter now clears all inference tensors before emptying the CUDA
cache. The same three-object process subsequently completed all 15 views,
which distinguishes the fixed lifecycle bug from a per-object VRAM limit.

A train-only source-view selection policy was also tested. It increased
held-out precision by `0.008415`, but reduced held-out IoU by `0.001502` and
recall by `0.003371`. The policy is retained as a negative experiment and was
rejected; using all four non-target views remains the selected fusion input.

A camera-space silhouette visual hull was then evaluated as a completion prior.
For every scored view, its target mask and dense point map are excluded while
the other four silhouettes are intersected in a streamed `96³` voxel grid.
The source-support threshold was selected only on validation objects
`ring_004`, `ring_012` and `ring_020`; requiring all four source silhouettes
gave `0.736988` validation mean IoU. Fixed-policy evaluation on all three test
objects, including the later `ring_015` extension, reached `0.738064` mean IoU,
`0.937174` mean recall and a `+0.272417` mean-IoU gain over the 518 px point-map
baseline. Every evaluated object improved.

The candidate is retained only as a silhouette-completion prior. Its minimum
test-view IoU is `0.415005`, so it still fails the geometry gate. Visual review
shows the expected failure: silhouettes become substantially complete, but
holes, concavities, relief, engraving, stone seats and component boundaries are
filled or absent. VGGT cameras were also inferred jointly from all five images,
so this is a leave-one-surface-out diagnostic, not a strict target-image
holdout. It must not be exported or described as completed CAD geometry.

The next checkpoint constrains that visual hull to remain near points observed
in the four source views. Seven shell distances from `1%` to `12%` of each
object's robust 3D diagonal were compared on the validation split. The selected
`4%` shell reaches `0.756890` validation mean IoU and raises the validation
minimum from `0.397525` to `0.565569`. Applying that fixed setting to all three
test objects gives `0.757313` test mean IoU, `0.503804` minimum, `0.840150`
precision and `0.879965` recall. This is `+0.019249` over the plain visual hull
and `+0.291666` over the 518 px point-map baseline.

Visual inspection confirms that source-surface support restores useful negative
space, especially the ring opening in top views. It can also cut an incorrect
or incomplete opening where point evidence is sparse, and one validation object
loses `0.003144` mean IoU versus the plain hull. The checkpoint is therefore
retained for further signed-depth/normal carving, not accepted as geometry.

Signed-depth free-space carving is the next retained checkpoint. It starts from
the four-source silhouette hull and selected surface shell, then removes voxels
that lie in front of the observed VGGT depth in too many source cameras. The
depth tolerance and allowed contradiction count were selected only on
`ring_004`, `ring_012` and `ring_020`. The selected `0.25%` robust-diagonal
tolerance with one allowed source contradiction reaches `0.780605` validation
mean IoU and `0.578221` minimum. Fixed-policy evaluation on `ring_007`,
`ring_015` and `ring_023` reaches `0.781514` mean IoU, `0.529146` minimum,
`0.875663` precision and `0.875165` recall. All three test objects improve over
the surface-aware hull, with a mean gain of `+0.024201`.

The target mask and target point map remain excluded from each reconstruction,
but cameras are jointly inferred from all five views. The geometry exit gate is
still false: surface normals, internal topology, relief and physical scale are
not validated. This evidence remains a research-only geometric prior.

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

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.validate_semantic_camera_priors

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.refine_vggt_semantic_cameras

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-geometry-vggt:a288dd0 \
  python -m pipeline.run_vggt_multiring_experiment \
  --checkpoint /workspace/models/vggt/VGGT-1B/model.pt \
  --image-size 518 --output-dir dataset/phase_runs/v1/vggt_resolution_518_v1

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.compare_vggt_resolution_ladder

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_vggt_view_transfer

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_vggt_visual_hull_completion

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_vggt_surface_aware_hull

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_vggt_signed_depth_carving
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
