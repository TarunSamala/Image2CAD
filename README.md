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

Phase 3.3 currently reaches mean silhouette IoU `0.8181`, detail IoU `0.7843` and boundary F1 `0.7952`. It is not labelled manufacturing-accurate: physical scale is uncalibrated and the evaluation masks are reviewed machine masks rather than human ground truth.

## Repository layout

```text
.
├── pipeline/       Phase extraction, reconstruction, refinement and validation
├── tests/          Regression tests for every retained phase
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

The current suite contains 28 tests.

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

## Artifact policy

- Keep source reference images and phase reports under version control.
- Keep accepted phase outputs versioned when they are needed for reproducibility.
- Do not commit downloaded checkpoints, Hugging Face caches, Python bytecode or runtime logs.
- Treat STEP as authoritative. STL/OBJ/GLB files are derived exchange or preview artifacts.
- Record rejected experiments before backtracking so unsuccessful directions remain traceable.

## Next acceptance requirement

Before Phase 3 can be approved as scale-aware, provide at least one known physical measurement, preferably gemstone diameter or inner ring diameter. Human-reviewed masks are also required for a defensible final visual score.
