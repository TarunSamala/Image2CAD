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
