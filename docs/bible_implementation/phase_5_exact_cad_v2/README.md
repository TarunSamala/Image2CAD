# Phase 5 - Exact CAD V2

This checkpoint starts the canonical Phase 5 CAD layer with reusable,
designed-truth generators. It does not infer dimensions from images and does
not claim that Ring01 or STL-1 has been reconstructed as exact CAD.

The project scope is now explicitly **rings only**. The generator may use
multiple ring topology families, but pendants, necklaces, idols and unrelated
object classes are excluded from this development path.

## Implemented foundation

- explicit `RingSpec` and per-stone `StoneSpec` contracts;
- independent shank, head bridge, stone, seat, cavity, prong and bezel solids;
- round and oval stone primitives;
- round-solitaire, oval-solitaire, three-stone and nine-stone halo fixtures;
- stable component IDs for every stone, seat, cavity, prong and bezel;
- one valid connected metal B-rep per fixture;
- separate valid gemstone solids;
- STEP export and independent STEP reload validation;
- watertight derived STL generation from authoritative exact solids;
- retained four-family visual audit.

All four fixtures pass the current structural checks. Together they contain 14
individually indexed stones, 14 seats, 14 cavities, 24 prongs and eight bezel
components.

## STL-1 image-evidence bridge

`pipeline/propose_phase5_stl1_ring_parameters.py` now processes all 24 STL-1
rings and all 120 prepared views. It records a normalized silhouette envelope
for every view and extracts the inner shank aperture and robust shank thickness
from each top view. The retained audit is
`data/phase5_stl1_ring_parameters_v1/stl1_ring_parameter_audit.png`.

The adapter preserves the 18/3/3 train, validation and test split. Its values
are expressed in ring units where the detected inner-aperture equivalent
diameter is 1.0. It deliberately leaves physical units and absolute dimensions
empty. These proposals may initialize render-and-compare fitting, but they are
not accepted CAD and cannot establish manufacturing accuracy.

## Phase 5.1 exact ring-body fitting

`pipeline/fit_phase5_stl1_ring_bodies.py` converts every detected shank into a
32-control periodic spline and extrudes it as an exact single-solid B-rep. All
24 objects export as STEP at nominal display scale. The training and validation
shanks are projected back into the source top view; the three test objects are
generated but deliberately remain unscored until the fitting policy is frozen.

The current train mean shank IoU is `0.932061` with boundary F1 `0.987267`.
Validation reaches `0.935501` IoU and `0.984843` boundary F1. These values
measure consistency with a machine-derived top-view shank target built from the
same silhouette; they are not independent CAD ground truth. The checkpoint
does not include shoulders, gallery, head, relief, stones or physical scale.

## Phase 5.2 exact head-support envelope

`pipeline/fit_phase5_stl1_head_supports.py` now adds a connected structural
head candidate to every Phase 5.1 shank. It intersects an exact extrusion of
the front ornament silhouette with an exact extrusion of the top head profile,
then joins two shoulder supports and verifies that the result remains one valid
metal B-rep. All 24 rings export as STEP; the three validation rings also retain
four-camera renders of the actual tessellated CAD so projection scores cannot
hide an obviously incorrect volume.

Training reaches `0.972574` mean front-profile IoU and `0.986451` mean top
envelope IoU. Validation reaches `0.973288` and `0.983941`. The first
bounding-box implementation reached only `0.515998` validation top IoU and was
rejected. A subsequent real-CAD preview exposed an inverted side-view aspect
ratio; that mapping was corrected before retaining this checkpoint.

These are same-silhouette fitting-consistency scores, not independent geometry
truth. The output is intentionally a filled visual-hull envelope: it does not
yet reconstruct relief, faces, openwork, stones, prongs, seats or cavities.
Left/right views provide disagreement diagnostics but are not calibrated well
enough to override the direct front silhouette.

## Phase 5.3 exact negative-space cut trials

`pipeline/fit_phase5_stl1_openwork.py` consumes the existing stable-ID Phase 4
front-view negative-space inventory instead of introducing another detector.
It trials all 168 candidates as exact through-cuts in their corresponding
Phase 5.2 head envelope. A cut is retained only when it intersects the head and
leaves one valid connected head solid; the complete fused ring metal must also
remain one valid solid.

All 168 candidates currently pass those geometric trial conditions. Training
front-view IoU rises from `0.972574` to `0.988620`; validation rises from
`0.973288` to `0.987284`, with `0.969656` validation boundary F1. All 24 STEP
exports and the validation CAD previews pass. The three test rings are generated
but remain unscored.

This result does not prove that every enclosed region is intentional openwork.
The source masks are machine proposals, candidate semantics remain unresolved,
and hidden cut depth cannot be observed from the front image. These are exact
negative-space CAD candidates for review—not confirmed stones, cavities or
manufacturing holes. Sculptural relief and deity/face detail remain absent.

## Phase 5.4 reversible relief evidence

`pipeline/build_phase5_stl1_relief_evidence.py` processes all 24 rings and all
120 prepared views. It does not map brightness directly to height. Instead it
requires image detail to agree with retained depth residuals, attenuates pixels
near silhouettes and openings, uses repeat uncertainty, and adds three-model
depth agreement for the 15 views where that supplement exists. Unsupported
highlights, shadows and texture are written to a separate rejected-detail map.

Every ring receives a front-view OBJ evidence surface at a 96-sample grid. The
OBJ is an open, reversible layer in nominal display units and is never fused
into the exact Phase 5.3 STEP body. All 120 computations are repeat
deterministic; the fixed test objects are generated but excluded from aggregate
selection metrics.

The retained validation result is intentionally conservative: mean confidence
is `0.044672`, only `0.076056` of thresholded fine detail is depth-supported,
and `0.238673` remains unresolved. Visual inspection of the actual OBJ previews
for rings 004, 012 and 020 confirms shallow broad-form relief, not faithful
deity-face reconstruction. This checkpoint is therefore evidence of the
remaining gap, not approval for CAD fusion.

## Phase 5.5 reciprocal relief-track consistency

`pipeline/evaluate_phase5_relief_tracks.py` reuses the retained 384-query VGGT
tracks for rings 001, 004, 012, 020, 007, 015 and 023. It does not rerun the
model, move cameras or deform geometry. For every source relief candidate it
requires target-view relief support, forward and reverse visibility/confidence,
a target-query distance below 12 pixels and a reciprocal-cycle error below 16
pixels. Normal agreement is reported separately for the 15 views where the
Metric3D direct-normal supplement exists, rather than changing the common gate.

The three validation rings retain 178 of 623 front candidates (`0.285714`).
Ring 004 retains only six, Ring 012 retains 121 and Ring 020 retains 51. The
isometric view carries most support; top views retain none because the ornament
face is largely not visible. The three test rings are rendered for evidence but
remain excluded from aggregate selection metrics.

This is sparse learned-feature evidence. VGGT saw all five images jointly,
nearest-query association is approximate, only seven rings had retained sparse
tracks at this checkpoint, and there is no physical correspondence truth. Phase 5.5 therefore
does not filter the OBJ permanently or authorize exact-CAD relief fusion.

## Phase 5.6 full-dataset reciprocal relief tracks

The retained VGGT extraction now covers all 24 STL-1 rings at 384 queries per
view. `evaluate_phase5_relief_tracks.py --all-stl1` applies the Phase 5.5 gate
without threshold changes. Training retains 1,420 of 4,610 front candidates
(`0.308026`); validation retains 178 of 623 (`0.285714`). Rings 007, 015 and
023 are generated for review but excluded from all aggregates. The run peaks
at about 2.62 GB CUDA memory on the 4 GB GPU.

Visual inspection confirms coherent front-to-isometric support, but support is
highly uneven and top-view reciprocity remains absent. These are sparse learned
tracks, not dense correspondence, physical truth or authorization to fuse
relief into the exact STEP body.

## Phase 5.7 no-regression relief-query densification

The fixed hybrid policy retains all 384 Phase 5.6 queries and their acceptance
decisions exactly, then appends 640 unique queries targeted at depth-supported
relief, silhouette boundaries and textured surfaces. The Phase 5.5 reciprocal
thresholds, cameras and CAD remain unchanged. This prevents denser sampling
from erasing earlier evidence while allowing additional regions to be tested.

Policy selection used all 18 training rings. Mean candidate-area coverage rose
by `0.357539`, mean reciprocally supported-area coverage rose by `0.159578`,
and every training ring improved. The frozen policy was then run once on rings
004, 012 and 020. Mean candidate-area coverage rose by `0.439941`, supported
coverage rose by `0.156563`, supported fraction rose by `0.030427`, and every
validation ring improved. All source queries were preserved with `0.0 px`
maximum source error, at least `0.999961` of tracks were in bounds, and peak
CUDA allocation remained about 2.62 GB. Rings 007, 015 and 023 remain untouched
by this experiment.

The validation audit was visually inspected and shows coherent additional
front-to-isometric and limited side-view support. Top-view support remains
nearly absent. Coverage improvement is therefore accepted as stronger review
evidence only; it does not validate dense correspondence, physical geometry,
metric scale, relief fusion or manufacturing accuracy.

## Phase 5.8 independent normal corroboration

All 15 views of the fixed validation rings now have retained outputs from Depth
Anything V2 Metric, Metric3D V2 and UniDepth V2. Metric3D supplies the direct
normal hypothesis. Its own depth output is excluded from the comparator, which
is built only from Depth Anything and UniDepth screen-space normals. The frozen
Phase 5.7 tracks and counts are reproduced exactly and are annotated, not
deleted or moved.

Of 736 front-source points supported by Phase 5.7, 521 (`0.707880`) also pass
the existing `0.40` normal-agreement threshold at source and target; 215 remain
unresolved. Per-ring corroboration is `49/72`, `331/453` and `141/211` for
rings 004, 012 and 020. Only one accepted front-to-top track exists across the
entire validation cohort, and it is corroborated. This confirms that missing
view overlap is now a larger limitation than query density.

The three-model depth stability score is `0.672911`, while Metric3D direct
normals differ from the three-depth consensus by `22.7153` degrees on average.
Visual inspection shows disagreement around fine carving, thin profiles and
head-to-shank transitions. Without normal ground truth or calibrated camera
rotations, Phase 5.8 remains an uncertainty annotation and cannot authorize
relief fusion, metric dimensions or manufacturing use.

## Phase 5.9 visibility-aware relief graph

`build_phase5_visibility_graph.py` converts the frozen Phase 5.7 reciprocal
tracks and Phase 5.8 normal annotations into explicit graph evidence for rings
004, 012 and 020. Each candidate front query receives a stable node ID. Every
accepted observation edge records the target view, target coordinate, reciprocal
cycle error, nearest-query distance, track score and normal status. A missing
edge stays missing; the pipeline no longer treats top or side views as if they
must observe a front-facing ornament surface.

The graph reproduces all 736 supported queries and all 521 normal-corroborated
queries. It contains 827 reciprocal observation edges, 567 normal-corroborated
edges and 30 deterministic image-space evidence regions. Only 45 supported
queries (`0.061141`) are normal-corroborated in two or more target views; 476
have one corroborated target, 215 remain reciprocal but normal-unresolved, and
1,797 source candidates have no accepted target view.

Visual inspection shows supported points following visible ornament regions,
but several regions remain broad because nearby points can share the same view
without being one semantic component. Region boxes therefore route later
component review; they are not CAD patches. The graph does not recover surfaces
hidden from the front view, prove calibrated angular diversity or authorize
relief fusion.

## Phase 5.10 visibility-to-component evidence bridge

`build_phase5_component_evidence_graph.py` gives each Phase 5.9 validation
region a stable relief-component ID and connects it to the existing exact metal
body and Phase 5.2 head-support candidate. A deterministic five-zone image-space
partition records location without pretending to identify jewellery semantics.
The bridge also records proximity to Phase 4 enclosed negative-space proposals.

The retained validation graph contains 30 relief components: three with
multi-view normal support, 22 with single-view normal support and five with
reciprocal tracks but unresolved normals. It links four relief components to
nearby negative-space evidence. The cohort audit was visually inspected; IDs
follow the retained visibility regions and the negative-space outlines are
spatially coherent. Some regions remain broad and contain multiple ornaments.

Every relief semantic label remains `unknown_ornamental_relief`. Negative-space
adjacency does not prove a gemstone, opening, seat or cavity, and spatial zones
do not prove a face or prong. This checkpoint establishes graph organization
only; semantic validation, hidden geometry and CAD fusion remain blocked.

## Phase 5.11 edge-depth-view relief subdivision

`subdivide_phase5_component_evidence.py` divides broad Phase 5.10 components
only where retained evidence supports a boundary. Queries remain connected when
they are spatially close, share a reciprocal target view, have continuous
repeat-regularized relative depth and do not cross a strong prepared-edge
barrier. A deterministic principal-axis split bounds transitive graph chains at
110 track pixels.

The first strict configuration produced 227 patches, including 163 singletons,
and was rejected after visual inspection as excessive fragmentation. The
retained configuration produces 83 patches from 30 parents. It partitions every
supported query exactly once, preserves 28 unresolved singleton observations
and marks 40 normal-supported patches with at least three queries for future
surface-fitting review. Fifteen patches contain multi-view normal support.

The retained cohort audit is visibly less noisy and prevents the largest Ring
012 and Ring 020 chains from spanning most of the ornament. Boundaries can still
follow reflections or depth-model bias, and a patch can split one ornament or
contain several. Every semantic label remains unknown, test rings remain
untouched and CAD geometry is unchanged.

## Phase 5.12 reversible local relief-surface trials

`build_phase5_local_relief_surfaces.py` triangulates only the 40 Phase 5.11
patches already marked eligible for fitting review. Each vertex keeps its front
image location and samples the conservative signed relief retained in Phase
5.4. Every patch is written as an independent open OBJ group, so a failed trial
can be removed without changing the exact ring body, head support or openwork
STEP checkpoints.

All 40 eligible validation patches generate finite geometry: 676 vertices and
1,018 faces in total. Mean same-view interpolation RMSE is `0.278`; Ring 004 is
the weakest object at `0.392`, compared with `0.205` for Ring 012 and `0.235`
for Ring 020. The cohort audit includes source-space triangles, normalized
interpolation error and renders of the actual OBJ artifacts. Visual inspection
confirmed that generated groups stay inside their eligible image regions, but
the surfaces are sparse and disconnected as expected.

These numbers measure agreement with the same image-depth evidence used to
construct the vertices. They do not establish physical relief accuracy,
semantic identity, metric scale or hidden geometry. Test objects are untouched,
no STEP is generated and CAD fusion remains blocked.

## Phase 5.13 independent target-view reprojection

`validate_phase5_local_relief_reprojection.py` keeps every Phase 5.12 OBJ
surface unchanged and projects its existing triangle connectivity through the
normal-corroborated reciprocal tracks retained in Phase 5.9. Source-derived
relief values are compared with separately prepared target-view relief. Target
evidence is never used to refit the source surface.

Across the three validation rings, 56 target-view evaluations are possible.
Only five pass the complete image-evidence gate. Five patches receive
single-target review support, none receives support from two targets, 26 are
target-inconsistent and nine remain unresolved because the sparse tracks cannot
form a projected triangle. Ring 004 has no passing target evaluation. Visual
inspection confirms that the projected triangles land on the expected target
ornaments, so the weak result is retained rather than hidden by relaxed gates.

Target coordinates remain VGGT hypotheses rather than calibrated projections,
and target relief comes from the same family of image-depth estimators. A pass
therefore supports review only. Physical accuracy, metric scale, semantics,
hidden geometry, surface promotion and CAD fusion all remain unvalidated.

## Phase 5.14 frozen-gate failure attribution

`diagnose_phase5_target_reprojection.py` reads the retained Phase 5.13 reports
without recomputing scores or changing thresholds. It assigns a primary blocker
to each target evaluation and creates a ranked CSV action queue for every local
surface patch.

Twelve evaluations are primarily blocked by insufficient normal-supported
tracks, 12 fail to transfer relief better than a flat baseline, eight have
orientation instability, eight exceed the target-relief RMSE limit, six cannot
form a stable target triangle and five have insufficient projected area. The
five passing evaluations are preserved. At patch level, 24 are routed to
cross-view track-topology research on the training split, 11 to view-invariant
relief research and five to finding a second independent target view.

These labels explain which frozen check failed first; they do not prove a
unique physical cause. The action queue is a research plan, not validation-set
tuning, geometric improvement or CAD authorization.

## Phase 5.15 training reciprocal-topology density

`evaluate_phase5_training_track_topology.py` implements the structural part of
the Phase 5.14 action queue on the 18 training rings only. It reuses the
existing reciprocal gate and preserves the first 384 baseline queries exactly.
For each target view, it forms a bounded local triangle only when all three
source vertices are accepted in that same target. Degenerate projections and
orientation flips are retained as failures rather than hidden.

The Phase 5.7 hybrid policy raises orientation-preserved local triangles from
`2,252` to `6,913`. All 18 training objects improve and none regresses. Visual
inspection shows that the retained triangles remain local to visible relief
regions rather than crossing the ring opening or image background.

This is not complete cross-view geometry. Front-to-top topology exists on only
two of 18 training rings (`24` retained triangles total), while the isometric
route contributes `5,229`. The selected policy is therefore frozen only for
one validation topology evaluation. No validation improvement, metric surface,
hidden geometry, CAD fusion or manufacturing accuracy is claimed.

## Phase 5.16 frozen validation topology transfer

The selected Phase 5.15 topology settings were applied once to Ring 004, Ring
012 and Ring 020 after verifying the frozen training report and implementation
hashes. Stable local triangles increase from `363` to `1,017`, and all three
validation rings improve with no per-ring regression. The cohort and all three
target overlays were visually inspected: the triangles remain concentrated on
visible ornament regions. Front-to-top topology is still `0` on all three
rings, and Ring 020 has no stable right-side triangle.

This measures correspondence density, not independent relief transfer. The
Phase 5.13 surface gate still passes only `5/56` evaluations and has no patch
supported by two target views. No relief surface or CAD component is promoted.

## Phase 5.17 training relief-descriptor discrimination

The Phase 5.14 appearance/depth action was tested on all 18 training rings
using the frozen reciprocal tracks. Signed relief was compared with unsigned
depth detail, photo detail and a depth-photo fusion. Each accepted target
location was scored against nearby displaced controls on the same ring. The
controls had to satisfy the same relief-confidence floor and similar photo
detail, so easy low-detail negatives could not inflate the result.

The retained comparison includes `3,259` control-eligible tracks. Mean paired
discrimination across rings is `0.607` for existing signed relief, `0.532` for
unsigned depth, `0.516` for photo detail and `0.547` for the fusion. No
replacement meets the prespecified training selection gate. Only two top-view
tracks have eligible controls, which is insufficient for a top-view descriptor
claim. The first, less controlled photo-detail result looked stronger; matching
the controls exposed that bias. No candidate is sent to validation, and the
Phase 5.13 surface gate remains unchanged.

These are tests against machine-proposed correspondences, not physical point
labels or measured relief. The next experiment needs a representation that
retains local shape across viewpoint and reflection changes, plus a separate
check that the same patch is actually visible in each compared view.

## Claim boundary

The dimensions are designed test parameters. The fixtures verify generator
topology and export behaviour only. They do not validate image-derived
dimensions, unseen-image family generalization, aesthetics, physical scale,
clearance rules or manufacturing accuracy.

The current head bridge is deliberately simple. General shoulders, galleries,
channels, split shanks and exact render-and-compare image fitting remain open.

## Reproduce

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.benchmark_phase5_exact_cad_v2

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_exact_cad_v2

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.propose_phase5_stl1_ring_parameters

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_stl1_ring_parameters

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.fit_phase5_stl1_ring_bodies

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_stl1_ring_body_fit

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.fit_phase5_stl1_head_supports

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_stl1_head_support_fit

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.fit_phase5_stl1_openwork

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_stl1_openwork_fit

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase5_stl1_relief_evidence

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_stl1_relief_evidence

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_phase5_relief_tracks

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_relief_tracks

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_phase5_relief_tracks --all-stl1

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_full_relief_tracks

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_dense_relief_pilot \
    tests.test_phase5_dense_relief_validation

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_normal_corroboration

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase5_visibility_graph

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_visibility_graph

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase5_component_evidence_graph

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_component_evidence_graph

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.subdivide_phase5_component_evidence

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_relief_subdivision

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase5_local_relief_surfaces

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_local_relief_surfaces

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.validate_phase5_local_relief_reprojection

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_target_reprojection

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.diagnose_phase5_target_reprojection

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_target_failure_attribution

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_phase5_training_track_topology

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_training_track_topology

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_phase5_validation_track_topology

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_validation_track_topology

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_phase5_training_relief_descriptors

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace:/workspace/pipeline \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase5_training_relief_descriptors
```

## Next gate

1. Develop a local shape or patch representation on training rings that beats
   the Phase 5.17 signed-relief baseline under matched controls. Check actual
   target visibility before requiring a front-to-top match. The Phase 5.13
   surface gate must gain independent two-target support before any promotion.
2. Resolve candidate semantics and cross-view opening identity before any
   retained cut is promoted to an approved cavity or manufacturing hole.
3. Fit general gallery generators across side and iso evidence without
   weakening exact-solid validity.
4. Add channel and split-shank topology fixtures.
5. Extend the implemented normalized parameter adapter to head, shoulder,
   gallery and component-track parameters with explicit uncertainty.
6. Render exact ring exports through accepted camera hypotheses and fit only against
   reviewed or explicitly machine-proposed evidence.
7. Validate on held-out image families before making a generalization claim.
