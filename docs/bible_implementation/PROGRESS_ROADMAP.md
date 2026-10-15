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
- Bounded source-camera image-plane alignment was tested with translation and
  isotropic scale searches fitted from source views only. Although those fits
  improved source reprojection by up to `0.048` IoU, every non-identity profile
  reduced pooled validation reconstruction IoU. Validation selected the
  original VGGT cameras, and the experiment was rejected without changing the
  signed-depth baseline.
- Physically consistent rigid source registration was then tested by applying
  each bounded SE(3) transform to a source point map and its camera together.
  ICP lowered source-cloud distance, but even the tiny bound reduced validation
  mean IoU from `0.780605` to `0.779059`; the medium bound fell to `0.768162`.
  Repeated and symmetric jewellery surfaces make unconstrained nearest-neighbour
  matches unreliable, so validation again retained the original geometry.
- The pinned VGGT track head then produced 96 sparse queries per view for seven
  rings while staying below 4 GiB of GPU memory. Bounds and source-query
  integrity checks passed, but they do not prove physical correspondence.
  Track-guided rigid fitting selected the strict profile after a `+0.000444`
  validation mean-IoU gain, then lost `-0.002421` on the fixed test split.
  Visual review found long-baseline matches collapsing onto repeated ornament,
  so this experiment is rejected and signed-depth carving remains active.
- A denser follow-up produced 384 queries per view for the same seven-ring
  cohort while remaining at `2.44 GiB` peak CUDA allocation. Reciprocal
  A-to-B-to-A filtering removed many ambiguous tracks, but every non-identity
  registration profile reduced validation IoU. Validation therefore selected
  identity before test scoring; test IoU remains the signed-depth baseline at
  `0.781514 / 0.529146`. Dense tracks are retained only as unreviewed evidence
  for future component relationships, not as geometry correction.
- Projection-noise analysis found that more than `99.9%` of retained voxels
  belong to the largest 3D component on average; the visible scan-line gaps are
  primarily coarse voxel-to-pixel rasterisation artifacts. Validation selected
  a `3×3` close only for raw projections with at least four foreground
  components. The frozen renderer raises test mean IoU from `0.781514` to
  `0.788982`, improves every test object, regresses no test view and leaves the
  weakest IoU unchanged at `0.529146`. It is retained for cleaner review only;
  the voxel geometry and geometry-gate decision are unchanged.

## Remaining implementation

Phase 4 now has a deterministic component-graph API and a Ring01 machine
proposal with stable IDs, provenance, uncertainty and explicit
stone-to-seat/cavity/prong dependencies. A synthetic ten-stone graph passes all
30 remove/resize/replace locality cases without changing unrelated node state.
This is a graph-state benchmark only: Ring01 has zero approved graph nodes, its
four prongs still share an aggregate mask, and no CAD Boolean was regenerated.
The Phase 4 exit gate therefore remains blocked.

1. Build reviewed semantic component identities, then group dense
   correspondence by component and view direction. Image-plane fitting, rigid
   ICP, sparse VGGT tracks and denser reciprocal-cycle registration have all
   been rejected for global geometry movement by the fixed held-out split.
2. Use pairwise camera-angle diagnostics, projected-surface precision and
   completion recall separately instead of treating IoU as camera accuracy.
3. Improve the current signed-depth minimum from `0.529` to at least `0.700`
   while raising mean IoU from `0.782` to at least `0.800`; the current method
   still fails the geometry gate.
4. Fuse consistent multi-view depth and independently estimated normals while
   retaining uncertain pixels.
5. Select a public-deployment-compatible independent normal comparator; DSINE
   remains blocked without a separate site/commercial licence.
6. Connect the component graph to independently editable stone, prong, seat,
   cavity and relief geometry generators.
7. Repeat the ten-stone locality benchmark on exported exact CAD, not only
   graph state.
8. Generate parametric CAD and refine it through render-and-compare validation.
9. Export editable STEP plus derived STL and 3MF artifacts with explicit validation reports.

The target is a high-fidelity, structurally editable reconstruction. Under the
images-only constraint, verified millimetres and manufacturing accuracy cannot
be claimed without a scale reference.
