# Image-to-CAD jewellery reconstruction roadmap

![Development progress and implementation roadmap](assets/image_to_cad_progress_roadmap.png)

Status snapshot: 14 October 2026.

This visual summarizes the active **images-only** execution profile. It does not
replace or renumber the canonical [Project Bible roadmap](../bible_v1_0/19_DEVELOPMENT_ROADMAP.md).

## Current position

- Reproducibility, dataset organization and automated validation are established.
- Image intake, segmentation, silhouettes, edges and component evidence are implemented.
- Four metric-depth models have been evaluated; three cover all 20 selected views and Depth Pro covers four representative front views.
- Same-view consensus and disagreement maps are available.
- Metric3D direct surface normals and three-model depth-derived normal proxies
  are retained for 20 selected views, with angular disagreement maps.
- Cross-view camera and geometry alignment through VGGT is now under active
  validation; its outputs remain non-metric machine hypotheses.
- The public-deployment audit leaves VGGT with its gated commercial checkpoint
  as the only eligible learned cross-view candidate; no backend is product-ready.
- A noncommercial Ring01 VGGT run completed locally at 224 px on the 4 GiB GPU,
  but its fused point cloud failed visual review because of ghosting and outliers.
- Three more STL-1 rings now have dense VGGT runs: one training object and two
  test objects. All ran below 4 GiB. Their original 224 px point-map
  leave-one-surface-out scores were `0.106–0.158` mean IoU; the target point
  map is excluded, but VGGT cameras are still jointly inferred from all views.
- External real-image coverage includes 206 images for Phase 1/2. These images
  cannot be used as multi-view geometry evidence without verified same-object
  identities and view semantics.
- Object-centric reframing was tested on the same three VGGT objects and
  rejected: cohort mean leave-one-view-out IoU decreased from `0.140` to
  `0.126`. The failed refinement and comparison preview are retained.
- VGGT's pairwise optical-axis relationships are within `7.745` degrees of the
  named-view priors on average, but the semantic holdout minimum is only
  `0.274`. Naive semantic camera snapping reduced mean completion IoU to
  `0.068`, so the original learned cameras remain the baseline.
- A `224–518` px VGGT resolution ladder was evaluated. The selected 518 px
  setting was chosen on three validation rings (`0.428` to `0.482` mean IoU)
  and raises test-ring mean completion IoU from `0.132` to `0.492`,
  precision from `0.468` to `0.844`, and recall from `0.155` to `0.541` while
  remaining below 3.13 GiB peak CUDA allocation.
- Sequential 518 px execution now releases inference tensors between objects;
  all three representative rings complete in one process. A train-learned
  source-view policy was rejected because its small precision gain reduced
  held-out IoU and recall.
- Camera-space visual-hull completion now intersects the four non-target
  silhouettes. Validation mean IoU is `0.737`; full three-object test mean IoU
  is `0.738`, a `+0.272` gain over the selected point-map baseline. The minimum
  test view is still only `0.415`, and inspection confirms that the hull fills silhouettes
  without reconstructing relief, holes, cavities or component topology.
- A validation-selected surface shell now combines that hull with VGGT source
  points. At the selected `4%` robust-diagonal distance, validation mean/minimum
  IoU reaches `0.757 / 0.566` and the complete three-object test result reaches
  `0.757 / 0.504`. Test mean gains `+0.019` over the plain visual hull and
  `+0.292` over the point-map baseline, but inferred openings remain uncertain.
- Signed-depth free-space carving now rejects voxels that lie clearly in front
  of source-view VGGT surfaces. The validation-selected setting reaches
  `0.781 / 0.578` mean/minimum IoU. Fixed evaluation on all three test rings
  reaches `0.782 / 0.529`, improving mean IoU by `+0.024` over the surface-aware
  hull and `+0.316` over the point-map baseline. Every test object improves,
  but the minimum remains below the geometry gate and topology is unvalidated.
- Source-view normal-consistency carving was tested using finite-difference
  normals from VGGT world-point maps. Validation selected a zero cosine
  threshold, so no normal-based deletion was accepted. The experiment is
  rejected until an independent, better-aligned normal estimator is available.
- Structural negative-space margins were tested on large enclosed source-mask
  regions. A validation-selected 4 px margin raised the weakest held-out IoU
  from `0.529` to `0.542`, but reduced held-out mean IoU from `0.782` to
  `0.777`. It is retained as aperture evidence, not as the default geometry.

## Remaining implementation

1. Improve cross-view camera/point alignment before adding more carving rules;
   the normal and aperture experiments show that downstream deletion cannot
   repair misregistered outer profiles.
2. Use pairwise camera-angle diagnostics, projected-surface precision and
   completion recall separately instead of treating IoU as camera accuracy.
3. Improve the current signed-depth minimum from `0.529` to at least `0.700`
   while raising mean IoU from `0.782` to at least `0.800`; the current method
   still fails the geometry gate.
4. Fuse consistent multi-view depth and independently estimated normals while
   retaining uncertain pixels.
5. Select a public-deployment-compatible independent normal comparator; DSINE
   remains blocked without a separate site/commercial licence.
6. Build the universal jewellery component and constraint graph.
7. Support independently editable stones, prongs, seats, cavities and relief geometry.
8. Generate parametric CAD and refine it through render-and-compare validation.
9. Export editable STEP plus derived STL and 3MF artifacts with explicit validation reports.

The target is a high-fidelity, structurally editable reconstruction. Under the
images-only constraint, verified millimetres and manufacturing accuracy cannot
be claimed without a scale reference.
