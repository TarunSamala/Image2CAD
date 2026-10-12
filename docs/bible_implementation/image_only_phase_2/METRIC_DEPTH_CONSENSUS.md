# Image-only Phase 2 metric-depth consensus

This experiment compares independent metric-depth hypotheses on the same masked
jewellery pixels. It includes all five Ring01 views and all five views from one
STL-1 train, validation and test object (`ring_001`, `ring_004`, `ring_007`).

Backends are run sequentially because the target RTX 3050 Laptop GPU has 4 GiB
VRAM. Raw float32 arrays are retained before normalization:

- Depth Anything V2 Metric Small, indoor/Hypersim checkpoint;
- UniDepth V2 ViT-S, research-only CC-BY-NC checkpoint;
- Metric3D V2 ViT-S, with the assumed focal length recorded;
- Depth Pro, run on CPU or an isolated GPU process due to its 1B-parameter size.

The comparison performs robust per-object normalization, pairwise shape
correlation, median absolute disagreement and pixelwise median consensus. Model
agreement is evidence of **stability**, not evidence of correct depth.

Preview colour conventions are explicit and intentionally different by panel:

- model depth and median consensus: **red = near, blue = far**;
- model disagreement: **red = high disagreement, blue = low disagreement**.

Metric arrays retain their native distance direction, where larger depth means
farther from the camera. Only the display transform uses
`1 - normalized_depth`; no stored depth or consensus values are inverted.

## Retained run

The retained experiment uses one immutable 20-view manifest:

- Ring01: front, side, top, angled and back;
- `ring_001`: train split, all five views;
- `ring_004`: validation split, all five views;
- `ring_007`: test split, all five views.

Depth Anything V2 Metric, UniDepth V2 and Metric3D V2 completed all 20 views
sequentially on the RTX 3050 Laptop GPU. Peak allocated GPU memory was about
303 MB, 767 MB and 742 MB respectively. Depth Pro completed four representative
front views on CPU in 214 seconds. Its container reached approximately 7.5 GiB
during the safety probe, so the remaining 16 views were deliberately not run on
this 15 GiB laptop.

The final retained report records:

- three backends with 20/20 coverage;
- Depth Pro with 4/20 coverage;
- mean normalized cross-model stability of about `0.679`;
- minimum normalized stability of about `0.573`;
- no validated physical scale, millimetre measurement or depth accuracy.

Ring01 is most consistent on the broad shank gradient. Disagreement remains
concentrated around the gemstone, prongs, head, polished boundaries and
reflections. The sculptural STL-1 objects show high disagreement around faces,
relief and thin side profiles. These pixels must be carried forward as uncertain
geometry rather than converted directly to CAD dimensions.

## Reproduce

Prepare the immutable input list:

```bash
python -m pipeline.run_metric_depth_consensus --prepare-only
```

Each backend writes under
`dataset/phase_runs/v1/image_only_phase2_metric_depth_consensus_v1/backends`.
After at least two backends complete, generate the report and preview sheets:

```bash
python -m pipeline.run_metric_depth_consensus
```

The retained Depth Pro scope was deliberately limited to:

```text
ring01_reference:front
ring_001:front
ring_004:front
ring_007:front
```

All model folders and checkpoint files are content-hashed in their backend
reports. Model weights remain under the gitignored `models/` cache. Reproducible
raw depth, confidence, normal and consensus `.npy` tensors are also gitignored;
the compact reports and inspected PNG review sheets remain versioned.

## Measurement boundary

The models may emit values labelled in metres, but jewellery-scale accuracy has
not been validated. The experiment has no known physical dimension, calibrated
camera, scan or paired CAD. It must not emit verified millimetres or
manufacturing tolerances. A future scale anchor can convert a stable normalized
shape into an absolute measurement hypothesis.

VGGT/MASt3R alignment is not complete. The retained COLMAP run formed no sparse
model, the available VGGT path lacks a usable approved checkpoint and exceeds
the 4 GiB GPU gate, and MASt3R is not installed. The current output therefore
compares same-view model hypotheses only; it is not cross-view metric fusion.
