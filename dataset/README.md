# Jewellery multi-view dataset

The available source set contains 24 rings with five renders per ring: front, top, isometric, left-side, and right-side. Source files remain unchanged under `STL-1/`.

## Prepared dataset

Run:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace image2cad-validation:local \
  sh -lc 'PYTHONPATH=pipeline python pipeline/prepare_jewellery_dataset.py'
```

`prepared_v1/` contains:

- 768×768 standardized images with consistent object framing;
- conservative jewellery silhouette pseudo-masks;
- OpenCV edge maps;
- one JSON metadata record per ring;
- a JSONL manifest;
- object-level train, validation, and test split files;
- an integrity and readiness report.

The fixed split contains 18 training rings, 3 validation rings, and 3 test rings. All five views of a ring always stay in the same split.

## Loader

```python
from jewellery_multiview_dataset import JewelleryMultiViewDataset

dataset = JewelleryMultiViewDataset("dataset/prepared_v1", split="train", as_torch=True)
sample = dataset[0]
print(sample["images"].shape)  # (5, 3, 768, 768)
```

## Ground-truth limitation

The source contains no STL/STEP/3DM mesh, CadQuery program, physical dimensions, calibrated cameras, component masks, or individual stone annotations. The prepared silhouette and edge outputs are pseudo-labels for experimentation, not human-reviewed evaluation truth.

The loader rejects `require_cad_target=True` so this image-only set cannot accidentally be presented as supervised image-to-CAD training data. It is suitable for multi-view representation learning, vision preprocessing, pseudo-label experiments, and qualitative reconstruction tests.

## Canonical Bible Phase 2 dataset

`paired_cad_v1/` is a separate, currently empty registry for the real paired
benchmark. Objects are accepted there only when they contain 8–12 guided
images, production STEP, physical measurements, a component graph and legal
provenance. STL-1 is not automatically promoted into that registry.

Validate it with:

```bash
PYTHONPATH=pipeline python pipeline/bible_phase_2_paired_benchmark.py
```

## Legacy-numbered dataset experiment

The versioned `phase_runs/v1` experiment predates the Project Bible phase
definitions. Its directory names are preserved for provenance. It uses a
66,229-parameter U-Net that fits within the 4 GB laptop GPU:

- Legacy Phase 1 extracts deterministic silhouette, edge, contour, hole, occupancy, and symmetry measurements for all 120 views.
- Legacy Phase 2 trains whole-jewellery foreground segmentation on 18 rings.
- Legacy Phase 2.2 selects the threshold on 3 validation rings and tests it once on 3 unseen rings.
- The held-out pseudo-silhouette result is IoU `0.977586`, Dice `0.988646`, and boundary F1 `0.999753`.
- Legacy Phase 3 and later dataset training are blocked because this source has no paired CAD, cameras, scale, or component-instance truth.

The evidence under `phase_runs/v1/phase1_human_review_v1/` is the current
Bible Phase 1 review manifest. The machine proposals above feed that review,
but do not become human ground truth automatically. STL-1 cannot satisfy Bible
Phase 2 because it has no paired production CAD or metric dimensions.

Green areas in the test review sheets are target/prediction agreement. Blue and grey fringes show disagreement. These metrics measure reproduction of the prepared pseudo-masks, not manufacturing accuracy or performance on real jewellery photographs.

## External real-image ring intake

`external_sources/registry.json` keeps OrnAsia and Segment Your Ring separate
from STL-1 with their licences, revisions, hashes and supervision limits.
`external_ring_v1/` contains the reproducible Phase 1/2 intake manifest, report
and compact previews. Raw downloads and per-image artifacts are ignored; the
evidence report and previews are versioned.

The current external intake is `review_required`: 206 images were processed,
but only one of ten human-labelled masks clears the strict mask gate, and all
196 OrnAsia outputs lack independent mask truth. This dataset therefore expands
real-image failure coverage; it does not prove CAD, depth or manufacturing
accuracy.

## Comparison previews

Generate the high-resolution comparison sheets with:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/build_dataset_comparison_previews.py'
```

The sheets under `phase_runs/v1/comparisons/` show all multi-view objects and splits, source-to-preprocessing changes, and held-out prediction errors. In the prediction sheet, green is agreement, red is prediction-only, and blue is target-only.

## Ring-style phase audits

Generate the Ring01-style audit set for every dataset object:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/build_dataset_phase_audits.py --device cuda'
```

The output contains one all-phase sheet and five detailed view audits per ring. Reference, legacy Phase 1, legacy Phase 2, and legacy Phase 2.2 columns contain real dataset artifacts. Legacy Phase 3 through Phase 3.3.2 are visibly marked as not generated because this dataset contains no paired CAD, calibrated scale, cameras, or component-instance ground truth.

## Experimental legacy Phase 3 exports

`phase_runs/v1/phase3/visual_hull/` contains one STL, one 3MF, one comparison image, and one validation report for each of the 24 rings. Shared-axis scale alignment improves cross-view consistency, but the files remain non-metric visual hulls. Do not use their nominal 3MF display size for jewellery manufacturing.
