# Image-driven jewellery reconstruction

This repository experiments with reconstructing editable 3D/CAD jewellery from multi-view reference images. The current validation asset is Ring01: a four-prong solitaire ring shown from front, side, top, angled and back views.

The implementation combines classical vision, segmentation, multi-view geometric reasoning, parametric CadQuery solids and render-and-compare validation. The original general-purpose CAD reconstruction code retained in the repository is documented in [Upstream research](docs/UPSTREAM_CADRILLE.md).

## Current status

| Stage | Purpose | Status |
| --- | --- | --- |
| Phase 1 | Reference-image preparation and geometric observations | Validated |
| Phase 2/2.2 | Jewellery, stone, setting, prong and shank masks | Validated machine extraction |
| Phase 3/3.2 | Coarse reconstruction and topology correction | Retained as checkpoints |
| Phase 3.3 | Exact-solid multi-view refinement | Numerical and topology targets pass |
| Phase 3.3.1 | Individually addressable smooth cubic claws | Valid connected B-rep; visual fit preserved |
| Phase 3.3.2 | Four-claw visibility correction | Reference-fitted inspection views validated |

Phase 3.3.1 currently reaches mean silhouette IoU `0.8197`, detail IoU `0.7874` and boundary F1 `0.7960`. It replaces the straight segmented prong approximation with four named, smooth cubic claw lofts while preserving the accepted Phase 3.3 image fit. It is not labelled manufacturing-accurate: physical scale is uncalibrated, the evaluation masks are reviewed machine masks rather than human ground truth, and the `0.90` research target has not been reached. Phase 3.3.2 corrects the misleading 45-degree inspection camera so all four existing claws remain distinct in oblique previews; it does not modify the accepted STEP geometry.

## Repository layout

```text
.
├── pipeline/       Phase extraction, reconstruction, refinement and validation
├── tests/          Regression tests for every retained phase
├── dataset/        Source jewellery views and prepared object-level dataset
├── data/
│   ├── ring01_reference_images/   Five source views
│   ├── ring01_phase*/             Immutable/iterative phase checkpoints
│   └── logs/                      Historical runtime logs
├── docs/           Workflow diagrams and project documentation
├── models/         Local model checkpoints (ignored by Git)
├── hf_cache/       Local Hugging Face cache (ignored by Git)
├── Dockerfile.cadrille            GPU experiment environment
└── Dockerfile                     Original research environment
```

Each phase writes to its own directory. New refinement work should create or resume a versioned phase instead of overwriting an earlier checkpoint.

## Validate the project

With the existing `cadrille-gpu` container running:

```bash
docker exec cadrille-gpu sh -lc \
  'cd /workspace && PYTHONPATH=pipeline python -m unittest discover -s tests -v'
```

The current suite contains 63 tests.

## Jewellery dataset

`dataset/STL-1` contains the only available training and test source: 24 jewellery objects with five views each. The preparation pipeline standardizes the images, extracts conservative jewellery masks and edge maps, and splits by object so views of the same item cannot leak between training and evaluation.

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/prepare_jewellery_dataset.py'
```

The generated `dataset/prepared_v1` split contains 18 training, 3 validation and 3 test objects. It is suitable for image preprocessing, silhouette, edge and multi-view representation experiments. It does not contain CAD geometry, metric dimensions, calibrated cameras or per-component labels, so supervised CAD reconstruction and quantitative 3D accuracy evaluation are deliberately disabled. See `dataset/README.md` for the format and loader example.

Run the versioned dataset phase experiment:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/train_dataset_phases.py --device cuda'
```

The `dataset/phase_runs/v1` checkpoint processes all 120 views in Phase 1, trains on the 18 training objects, selects its threshold using only the validation objects, and evaluates 15 views from three unseen test objects. It reaches test pseudo-silhouette IoU `0.9776`. Dataset-wide progress stops honestly at Phase 2.2: Phase 3 through Phase 3.3.2 require CAD targets, calibrated cameras, physical scale, component instances, or an existing validated CAD model that this dataset does not provide.

## Rebuild Phase 3.3

```bash
docker exec cadrille-gpu sh -lc '
  cd /workspace &&
  PYTHONPATH=pipeline python pipeline/build_phase3_review_masks.py &&
  PYTHONPATH=pipeline python pipeline/refine_phase3_3.py --resume --rounds 5 &&
  PYTHONPATH=pipeline python pipeline/build_phase3_3.py &&
  PYTHONPATH=pipeline python pipeline/validate_phase3_3.py
'
```

Important outputs:

- `data/ring01_phase3_3/ring01_phase3_3.step` — authoritative editable assembly
- `data/ring01_phase3_3/ring01_phase3_3.stl` — derived watertight print mesh
- `data/ring01_phase3_3/phase3_3_validation.json` — strict validation report
- `data/ring01_phase3_3/comparisons/` — per-view visual comparisons
- `data/ring01_phase3_3/EXPERIMENTS.md` — accepted and rejected experiments

## Rebuild Phase 3.3.1

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace image2cad-validation:local sh -lc '  PYTHONPATH=pipeline python pipeline/refine_phase3_3_1.py &&  PYTHONPATH=pipeline python pipeline/build_phase3_3_1.py &&  PYTHONPATH=pipeline python pipeline/validate_phase3_3_1.py'
```

The authoritative editable output is `data/ring01_phase3_3_1/ring01_phase3_3_1.step`. The validation report and five reference comparisons are in the same checkpoint directory.

## Validate Phase 3.3.2 visibility

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace image2cad-validation:local sh -lc 'PYTHONPATH=pipeline python pipeline/correct_phase3_3_2.py'
```

The corrected four-claw preview and validation report are in `data/ring01_phase3_3_2/`. Phase 3.3.2 references the authoritative Phase 3.3.1 STEP instead of duplicating the 3D exports.

## Artifact policy

- Keep source reference images and phase reports under version control.
- Keep accepted phase outputs versioned when they are needed for reproducibility.
- Do not commit downloaded checkpoints, Hugging Face caches, Python bytecode or runtime logs.
- Treat STEP as authoritative. STL/OBJ/GLB files are derived exchange or preview artifacts.
- Record rejected experiments before backtracking so unsuccessful directions remain traceable.

## Next acceptance requirement

Before Phase 3 can be approved as scale-aware, provide at least one known physical measurement, preferably gemstone diameter or inner ring diameter. Human-reviewed masks are also required for a defensible final visual score.
