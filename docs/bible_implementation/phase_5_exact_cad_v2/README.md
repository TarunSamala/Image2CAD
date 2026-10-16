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
```

## Next gate

1. Densify Phase 5.6 reciprocal support and add a public-deployment-compatible
   independent normal comparator before promoting relief into exact CAD.
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
