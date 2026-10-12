# Image-only surface-normal evidence

This checkpoint continues the Bible camera/depth/normal benchmark using only
retained images and model outputs. It does not require CAD, scans, dimensions or
camera calibration.

## Evidence used

- Metric3D V2 direct per-pixel normal predictions for all 20 selected Ring01 and
  STL-1 views.
- Screen-space normal proxies derived separately from the retained Depth
  Anything V2 Metric, Metric3D V2 and UniDepth V2 depth maps.
- Per-pixel angular disagreement between the three depth-derived proxies.
- Per-pixel angular difference between the oriented Metric3D direct prediction
  and the depth-derived consensus.

The screen-space proxy uses masked percentile normalization before calculating
image gradients. It captures local orientation evidence, but it is not a metric
camera-space normal because camera intrinsics and physical scale are unknown.

## Retained result

The checkpoint covers 20 views across Ring01 plus one train, validation and
held-out STL-1 object. The report and previews are stored under:

```text
dataset/phase_runs/v1/surface_normal_evidence_v1/
```

The mean disagreement among depth-derived normal proxies is approximately
`7.95°`. The mean difference between Metric3D's direct normal prediction and the
depth-derived consensus is approximately `19.63°`. These are model-consistency
measurements, not accuracy against true surface normals.

The inspected previews show that broad shank and body orientation is generally
coherent. Disagreement concentrates around silhouettes, prongs, stones,
negative spaces and high-detail sculptural relief—the same regions where
reflection, refraction and depth discontinuities make image-only inference less
reliable.

## DSINE position

[DSINE](https://github.com/baegwangbin/DSINE) remains the planned independent
normal comparator. Its code and checkpoint are not silently downloaded by this
checkpoint. A future run must pin the source revision, checkpoint hash, licence
evidence, camera-intrinsic assumptions and runtime before its output can enter
the retained comparison.

## Claims and limits

This checkpoint establishes that per-pixel normal hypotheses and disagreement
maps are available. It does not establish:

- surface-normal accuracy;
- calibrated camera-space orientation;
- cross-view normal fusion;
- metric dimensions;
- manufacturing accuracy.

Polished metal, gemstones, shadows and highlights can influence every model in
the comparison. These regions remain uncertainty evidence for later camera and
component-graph work.

## Rebuild

```bash
python -m pipeline.run_surface_normal_evidence --force
```

The command uses retained depth outputs and does not run a GPU model.
