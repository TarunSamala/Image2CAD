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
```

## Next gate

1. Replace the filled head envelope with separate base relief, openwork and
   structural layers while preserving the accepted outer support envelope.
2. Fit general gallery generators across side and iso evidence without
   weakening exact-solid validity.
3. Add channel and split-shank topology fixtures.
4. Extend the implemented normalized parameter adapter to head, shoulder,
   gallery and component-track parameters with explicit uncertainty.
5. Render exact ring exports through accepted camera hypotheses and fit only against
   reviewed or explicitly machine-proposed evidence.
6. Validate on held-out image families before making a generalization claim.
