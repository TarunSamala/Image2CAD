# External ring image sources

This directory records third-party datasets without mixing them into `STL-1`.
The source registry is versioned; downloaded archives and raw images are local,
re-downloadable material and are excluded from Git.

## Roles

- **OrnAsia** is evaluation-only until physical-object identities are available.
  Multiple photographs of the same ring could otherwise leak across a random
  train/test split.
- **Segment Your Ring** retains its published eight-image training and two-image
  test split. Its masks supervise only the complete ring silhouette.

Neither source provides camera calibration, metric scale, depth truth, meshes,
production CAD, stones, prongs, seats, or cavity labels.

## Intake

After downloading and extracting both sources under `raw/`, run:

```bash
PYTHONPATH=pipeline python pipeline/intake_external_ring_datasets.py \
  --ornasia-root dataset/external_sources/raw/ornasia \
  --syr-root dataset/external_sources/raw/segment_your_ring \
  --strict-counts
```

The command generates a provenance manifest, Phase 1/2 image evidence, compact
comparison previews, and an intake report under `dataset/external_ring_v1/`.

## Current measured result

All 206 images decode and retain unique source hashes. Segment Your Ring's ten
human masks produce a mean refined IoU of `0.734473` and boundary F1 of
`0.308597`. Under the strict mask gate (IoU `0.80`, boundary F1 `0.50`), only
one mask is approved. All OrnAsia masks remain review-required because the
source has no human masks and the cloth background defeats plain background
differencing.

`audit_jewellery.py` now has an experimental small-object SAM prompt fallback.
It first finds compact high-detail jewellery, rejects long background folds,
and uses a negative centre prompt when local image evidence indicates an open
ring. This improves representative OrnAsia samples, but it is not batch-approved
without human masks. SAM confidence and agreement between machine phases are
recorded as diagnostics only, never as accuracy.
