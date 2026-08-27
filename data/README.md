# Data and generated artifacts

The Ring01 workflow keeps every major phase in a separate directory so results can be compared or backtracked without overwriting earlier geometry.

## Active Ring01 files

| Path | Role |
| --- | --- |
| `ring01_reference_images/` | Normalized front, side, top, angled and back source images |
| `ring01_artifact.json` | Prepared inference artifact |
| `ring01_metadata.json` | Extracted jewellery metadata |
| `ring01_output.py` | Generated CadQuery program |
| `ring01_output_512.py` | Historical 512-token generated program |
| `ring01_ring.step` / `.stl` | Early generated reconstruction retained for comparison |
| `ring01_phase1/` | Initial vision measurements |
| `ring01_phase2*` | Segmentation and feature extraction checkpoints |
| `ring01_phase3*` | Reconstruction, refinement and CAD validation checkpoints |
| `ring01_validation*` | Cross-phase validation artifacts |
| `logs/` | Historical command output; new logs are ignored by Git |

## Phase-directory convention

Each phase directory should contain its own reports, masks/renders, comparisons and exported geometry. New work should use a new versioned directory or an explicit resumable checkpoint.

Do not silently replace an accepted earlier phase. Record experiments and validation results so a change can be evaluated and backtracked.

## Authoritative formats

- Source reference images are the visual input.
- JSON reports are the machine-readable validation record.
- STEP is the authoritative editable CAD representation.
- STL, OBJ and GLB are derived print, exchange or preview formats.
- Preview and comparison images support inspection but are not measurement ground truth.

Information about datasets used by the original general-purpose research code is retained in [Upstream datasets](../docs/UPSTREAM_DATASETS.md).
