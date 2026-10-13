# Image-to-CAD jewellery reconstruction roadmap

![Development progress and implementation roadmap](assets/image_to_cad_progress_roadmap.png)

Status snapshot: 12 October 2026.

This visual summarizes the active **images-only** execution profile. It does not
replace or renumber the canonical [Project Bible roadmap](../bible_v1_0/19_DEVELOPMENT_ROADMAP.md).

## Current position

- Reproducibility, dataset organization and automated validation are established.
- Image intake, segmentation, silhouettes, edges and component evidence are implemented.
- Four metric-depth models have been evaluated; three cover all 20 selected views and Depth Pro covers four representative front views.
- Same-view consensus and disagreement maps are available.
- Metric3D direct surface normals and three-model depth-derived normal proxies
  are retained for 20 selected views, with angular disagreement maps.
- Cross-view camera and geometry alignment through MASt3R/VGGT remains the next gate.
- The public-deployment audit leaves VGGT with its gated commercial checkpoint
  as the only eligible learned cross-view candidate; no backend is product-ready.
- A noncommercial Ring01 VGGT run completed locally at 224 px on the 4 GiB GPU,
  but its fused point cloud failed visual review because of ghosting and outliers.
- Three more STL-1 rings now have dense VGGT runs: one training object and two
  held-out test objects. All ran below 4 GiB, but all failed leak-free
  leave-one-view-out reprojection (`0.106–0.158` mean IoU).
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

## Remaining implementation

1. Add model-aware visible-surface fusion and hidden-surface completion; do not
   generate CAD from the current partial clouds.
2. Use pairwise camera-angle diagnostics, projected-surface precision and
   completion recall separately instead of treating IoU as camera accuracy.
3. Evaluate a resolution ladder on the representative cohort only after the
   revised diagnostics show that added pixels can address the remaining error.
4. Fuse consistent multi-view depth and normals while retaining uncertain pixels.
5. Select a public-deployment-compatible independent normal comparator; DSINE
   remains blocked without a separate site/commercial licence.
6. Build the universal jewellery component and constraint graph.
7. Support independently editable stones, prongs, seats, cavities and relief geometry.
8. Generate parametric CAD and refine it through render-and-compare validation.
9. Export editable STEP plus derived STL and 3MF artifacts with explicit validation reports.

The target is a high-fidelity, structurally editable reconstruction. Under the
images-only constraint, verified millimetres and manufacturing accuracy cannot
be claimed without a scale reference.
