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

## Remaining implementation

1. Add DSINE as a pinned independent normal comparator.
2. Recover camera poses and reliable cross-view correspondences.
3. Fuse consistent multi-view depth and normals while retaining uncertain pixels.
4. Build the universal jewellery component and constraint graph.
5. Support independently editable stones, prongs, seats, cavities and relief geometry.
6. Generate parametric CAD and refine it through render-and-compare validation.
7. Export editable STEP plus derived STL and 3MF artifacts with explicit validation reports.

The target is a high-fidelity, structurally editable reconstruction. Under the
images-only constraint, verified millimetres and manufacturing accuracy cannot
be claimed without a scale reference.
