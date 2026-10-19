# Image-driven jewellery reconstruction

This repository experiments with reconstructing editable 3D/CAD jewellery from multi-view reference images. The current validation asset is Ring01: a four-prong solitaire ring shown from front, side, top, angled and back views.

The implementation combines classical vision, segmentation, multi-view geometric reasoning, parametric CadQuery solids and render-and-compare validation. The original general-purpose CAD reconstruction code retained in the repository is documented in [Upstream research](docs/research/UPSTREAM_CADRILLE.md).

## Which phase system is current?

The **Project Bible v1.0 roadmap is canonical**. It defines Phase 0 through
Phase 9. Earlier Ring01 and STL-1 experiments also used labels such as
Phase 1, Phase 2 and Phase 3.3.2, but those labels are now explicitly retained
as the **legacy pipeline** and must not be used to describe Bible completion.

| System | Meaning | Authority |
| --- | --- | --- |
| Bible Phase 0–9 | Current development, evidence and release gates | **Canonical** |
| Legacy Phase 1–3.3.2 | Historical preprocessing, segmentation and Ring01 CAD checkpoints | Retained evidence only |

Start with the [documentation index](docs/README.md), the
[canonical Bible roadmap](docs/bible_v1_0/19_DEVELOPMENT_ROADMAP.md), or the
[phase registry](docs/phase_registry.json). New work must use Bible phase names.

## Bible roadmap status

| Bible stage | Purpose | Current status |
| --- | --- | --- |
| Phase 0 | Reproducibility freeze | Complete |
| Phase 1 | Human-reviewed image evidence and ground truth | Machine evidence complete for Ring01 and all 24 STL-1 rings; human review and metric input pending |
| Phase 2 | Real paired photo-to-production-CAD benchmark | Not applicable under the images-only input constraint; retained as an unavailable validation gate |
| Image-only Phase 2 | Reflection-resilient evidence and multi-view consistency | In progress; Ring01 angled inner-shank correction proposed |
| Phase 3 | Camera, depth, normal and geometry evidence benchmark | Partial research evidence only; exit benchmark not complete |
| Phase 4 | Component and constraint graph | Partial; synthetic exact-CAD locality passes 30/30, while reviewed image-derived component truth remains blocked |
| Phase 5 | Exact CAD V2 | Phase 5.13 passes only 5/56 independent target gates. Phase 5.15–5.16 improve image-space topology on training and validation, while top-view topology remains absent on validation. Phase 5.17 rejects three relief-descriptor replacements after matched-control training tests. Surfaces remain review evidence separate from STEP |
| Phase 6–9 | Detail, manufacturing and pilot | Not started |

## Legacy checkpoint status

| Legacy stage | Purpose | Status |
| --- | --- | --- |
| Phase 1 | Image preprocessing and feature proposals | Validated machine extraction |
| Phase 2/2.2 | Jewellery, stone, setting, prong and shank mask proposals | Validated machine extraction |
| Phase 2.3 | Detail evidence and semantic-review proposal | Implemented; review pending |
| Phase 3/3.2 | Coarse reconstruction and topology correction | Retained as checkpoints |
| Phase 3.3–3.3.2 | Ring01 exact-solid refinement and four-claw visibility correction | Retained validated Ring01 checkpoint |

Phase 3.3.1 currently reaches mean silhouette IoU `0.8197`, detail IoU `0.7874` and boundary F1 `0.7960`. It replaces the straight segmented prong approximation with four named, smooth cubic claw lofts while preserving the accepted Phase 3.3 image fit. It is not labelled manufacturing-accurate: physical scale is uncalibrated, the evaluation masks are reviewed machine masks rather than human ground truth, and the `0.90` research target has not been reached. Phase 3.3.2 corrects the misleading 45-degree inspection camera so all four existing claws remain distinct in oblique previews; it does not modify the accepted STEP geometry.

## Repository layout

```text
.
├── pipeline/       Bible implementations, shared tools and legacy modules
├── tests/          Regression tests grouped by phase system
├── dataset/        Source jewellery views and prepared object-level dataset
├── data/
│   ├── ring01_reference_images/   Five source views
│   ├── ring01_phase*/             Immutable/iterative phase checkpoints
│   └── logs/                      Historical runtime logs
├── docs/
│   ├── bible_v1_0/          Canonical specification and roadmap
│   ├── bible_implementation/ Current implementation evidence
│   ├── legacy_pipeline/     Historical phase documents and diagrams
│   └── research/            Upstream and model research
├── models/         Local model checkpoints (ignored by Git)
├── hf_cache/       Local Hugging Face cache (ignored by Git)
├── Dockerfile.cadrille            GPU experiment environment
└── Dockerfile                     Original research environment
```

Each phase writes to its own directory. Existing artifact paths remain stable
for provenance. New development uses the Bible roadmap and must not overwrite
an earlier checkpoint.
See [Repository structure](docs/REPOSITORY_STRUCTURE.md) for the dataset-run layout and file-placement conventions.
See [Jewellery Phase Auditor](docs/legacy_pipeline/UPLOAD_AUDITOR.md) for the historical one-image and five-view upload program.
See [Lalitha Studio](docs/bible_implementation/STUDIO.md) for the Bible-aligned local evidence browser, phase comparisons, hard-gate status and research artifact downloads.
See [STL-1 Bible Phase 1 human review](docs/bible_implementation/phase_1_ground_truth/STL1_HUMAN_REVIEW.md) for the 24-ring, 120-view review contract and correction workflow.
See the [legacy phase workflow](docs/legacy_pipeline/images/image2cad-phase-flowchart-v1.png) for the earlier visual pipeline.

Phase 2.3 adds category-independent internal-edge, ridge, valley, relief, negative-space and reflection evidence. Every local detail proposal receives an addressable observation ID and an editable review decision. Cross-view IDs remain hypotheses until reviewed or confirmed by calibrated geometry; the stage does not automatically label bright regions as gemstones.

## Validate the project

With the existing `cadrille-gpu` container running:

```bash
docker exec cadrille-gpu sh -lc \
  'cd /workspace && PYTHONPATH=pipeline python -m unittest discover -s tests -v'
```

The current suite contains 148 tests.

## Active image-only Phase 2

The project accepts reference images only. Ring01 reflection-sensitive evidence
is corrected as a versioned, uncertainty-marked machine proposal:

```bash
PYTHONPATH=pipeline python pipeline/image_only_reflection_correction.py
```

The original paired-CAD validator remains as an inactive scientific guardrail;
it is not an input request for the active project.

## Retained paired-data validation gate

The canonical paired registry remains intentionally empty because production
CAD and measurements are unavailable:

```bash
PYTHONPATH=pipeline python pipeline/bible_phase_2_paired_benchmark.py
```

The command validates 8–12 views, camera metadata, production STEP, physical
measurements, component graph, object-level split and legal provenance. See the
[Phase 2 contract](docs/bible_implementation/phase_2_paired_benchmark/README.md).

## Jewellery dataset

`dataset/STL-1` contains the only available training and test source: 24 jewellery objects with five views each. The preparation pipeline standardizes the images, extracts conservative jewellery masks and edge maps, and splits by object so views of the same item cannot leak between training and evaluation.

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/prepare_jewellery_dataset.py'
```

The generated `dataset/prepared_v1` split contains 18 training, 3 validation and 3 test objects. It is suitable for image preprocessing, silhouette, edge and multi-view representation experiments. It does not contain CAD geometry, metric dimensions, calibrated cameras or per-component labels, so supervised CAD reconstruction and quantitative 3D accuracy evaluation are deliberately disabled. See `dataset/README.md` for the format and loader example.

Run the versioned **legacy-numbered** dataset phase experiment:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/train_dataset_phases.py --device cuda'
```

The `dataset/phase_runs/v1` checkpoint processes all 120 views in legacy Phase 1, trains on the 18 training objects, selects its threshold using only the validation objects, and evaluates 15 views from three unseen test objects. It reaches test pseudo-silhouette IoU `0.9776`. Legacy dataset-wide progress stops honestly at Phase 2.2. These artifacts are reused as proposals for Bible Phase 1 human review; they do not complete Bible Phase 2.

## Rebuild legacy Ring01 Phase 3.3

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

## Rebuild legacy Ring01 Phase 3.3.1

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace image2cad-validation:local sh -lc '  PYTHONPATH=pipeline python pipeline/refine_phase3_3_1.py &&  PYTHONPATH=pipeline python pipeline/build_phase3_3_1.py &&  PYTHONPATH=pipeline python pipeline/validate_phase3_3_1.py'
```

The authoritative editable output is `data/ring01_phase3_3_1/ring01_phase3_3_1.step`. The validation report and five reference comparisons are in the same checkpoint directory.

## Validate legacy Ring01 Phase 3.3.2 visibility

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

## Validate a real-photo sample

Run Phase 1 normalization and Phase 2 silhouette, shadow, edge, and support refinement on the validation-only sample:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/validate_sample_ring.py --device cuda'
```

The sample has one uncalibrated view, so its outputs are qualitative machine proposals. The four outer structures are recorded as support proposals rather than confirmed gemstone prongs, and pixel accuracy cannot be scored without a human-reviewed mask.

## Legacy dataset Phase 3 visual hulls

Generate experimental non-metric STL and 3MF visual hulls for all 24 five-view rings:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:local sh -lc \
  'PYTHONPATH=pipeline python pipeline/reconstruct_dataset_phase3.py --device cuda'
```

The reconstruction first aligns shared X, Y, and Z silhouette extents across independently normalized views, then carves a 128-voxel visual hull. The outputs are watertight research meshes and are explicitly named `non_metric`. They are not editable parametric CAD or manufacturing geometry: hidden concavities, semantic components, camera calibration, and physical scale remain unavailable. Phase 3.3 is therefore not claimed.

## Browse STL-1 in Lalitha Studio

The Bible-aligned Flask interface reads existing versioned evidence and records explicit Phase 1 human-review decisions for Ring01 and all 24 STL-1 rings. Corrected silhouettes are stored separately from immutable proposals. It never starts training or reconstruction from a browser request:

```bash
python -m pip install -r requirements-studio.txt
python -m studio.app
```

Then open `http://127.0.0.1:8501`. The UI includes all 24 five-view objects, Phase 1/2 evidence, experimental Phase 3 previews, Bible hard gates, non-metric STL/3MF downloads and per-object provenance reports. It classifies this dataset as Tier A and keeps the final decision at `research_only`.
