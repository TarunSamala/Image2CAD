# Dataset phase run v1

This directory is the reproducible run for `dataset/prepared_v1`.

- Root JSON files summarize phase coverage, Phase 1 extraction, Phase 2 training, and Phase 2.2 held-out evaluation.
- `checkpoints/` contains the trained segmentation checkpoint.
- `comparisons/` contains dataset-wide comparison sheets.
- `phase_audits/` contains per-object phase comparisons.
- `test_predictions/` contains held-out prediction images.
- `phase3/visual_hull/` contains one non-metric STL, 3MF, preview, and validation report per object.

The Phase 3 exports preserve image-derived shape evidence but do not contain calibrated physical scale or manufacturing geometry.
