# Lalitha Studio — STL-1 Evidence Console

Lalitha Studio is a lightweight Flask evidence console for `dataset/STL-1`, its dataset-wide Phase 1 review workspace and the retained Ring01 review. Source and machine artifacts are read-only. Accepted writes are identified human decisions and separate corrected silhouette masks. GPU inference is never run inside a web request.

## What it shows

- A focused computer-vision evidence index for normalization, shape, detail,
  depth and uncertainty extraction.
- All 24 rings and their five source views.
- The normalized 768 px images, pseudo-silhouettes and OpenCV edge maps.
- Per-view legacy Phase 2 IoU, Dice and boundary-F1 measurements.
- The retained legacy Phase 1 to Phase 2.2 audit sheet for every object.
- Active image-only Phase 2 Ring01 reflection correction, uncertainty and audit.
- The original paired-CAD gate marked unavailable under the image-only input constraint.
- Held-out prediction sheets for the three test objects.
- Experimental Phase 3 visual-hull previews and validation metadata.
- Direct downloads of each non-metric STL and 3MF artifact.
- A portable per-object JSON evidence report.
- Side-by-side Ring01 source/mask review with named decisions for labels, component identities and view semantics.
- Dataset-wide review for all 24 STL-1 rings, 120 views and 840 evidence items.
- Per-ring cross-view inventories for visible stones, prongs and sculptural relief.
- Versioned corrected silhouettes without overwriting machine proposals.
- Dataset-wide Depth Anything V2 evidence and selected-view agreement from
  Depth Anything V2 Metric, UniDepth V2, Metric3D V2 and Depth Pro.
- Surface-normal hypotheses from Metric3D and depth-derived normal consensus,
  including per-pixel angular disagreement for selected Ring01/STL-1 views.
- External-ring generalisation evidence, provenance and review state.
- A restrained black, white and grey interface, with pink reserved for focus,
  warnings and selected evidence.
- Explicit depth-preview colour semantics: red is near and blue is far for
  depth/consensus; blue is low and red is high for model disagreement.

The interface labels Phase 3 correctly: these are watertight research meshes at nominal display scale. The dataset has no calibrated cameras, physical dimensions, component ground truth or CAD targets, so the outputs are not manufacturing-validated CAD.

## Project Bible alignment

This interface follows `BASELINE ARCHITECTURE V1.0`:

- The current five-view dataset is classified as **Tier A — exploratory**.
- Source images remain immutable observed evidence.
- Normalization, pseudo-masks, edges and visual hulls are derived/inferred evidence.
- Legacy Phase 2 IoU is labelled pseudo-label self-consistency, not human-ground-truth accuracy.
- Image-only Phase 2 is displayed separately from legacy pseudo-mask metrics.
- The original Bible paired-data gate is retained as unavailable and does not
  request STEP, measurements, scans or calibration from the user.
- Phase 3 STL/3MF files are research-only derived meshes, not authoritative CAD.
- STEP/exact B-rep remains the required authoritative functional-geometry output.
- Metric, camera, reviewed-component, exact-B-rep and manufacturing gates are shown as blocked when their required evidence is absent.
- Every exported Studio report uses the final decision `research_only` and records which benchmark provenance fields are still missing.

The Studio belongs to the Bible's local/deterministic deployment boundary. It reads evidence, reports provenance and exposes review state. It does not let a browser request run a GPU model, alter source images or silently write inferred geometry into STEP. It can update explicit Ring01/STL-1 review records and store a corrected STL-1 silhouette separately from its immutable proposal.

The frozen development order remains:

```text
source-control reproducibility → human image-space ground truth → camera/depth/normal evidence → reviewed component and constraint graph → exact CAD generalization

The Component Graph tab exposes the Phase 4 Ring01 proposal, stable IDs,
provenance and manufacturing roles. It also provides non-persistent synthetic
ten-stone edit previews. Those previews test graph dependencies only and never
generate or modify CAD. The same tab compares aggregate prong masks with
separate front/top/angled proposals and explicit side/back overlap groups. It
also displays the retained VGGT normalized centroid-distance matrices used to
check whether multi-view geometry supports the proposed stable IDs.
The 24-object STL-1 front-view audit adds aperture and openwork proposals from
all 120 views. These remain negative-space candidates, not confirmed stones,
so the UI keeps semantic and CAD-authorization gates closed.
```

## Run locally

From the repository root, use a Python environment with the small Studio requirement:

```bash
python -m pip install -r requirements-studio.txt
python -m studio.app
```

Open <http://127.0.0.1:8501>.

Optional settings:

```bash
IMAGE2CAD_STUDIO_HOST=0.0.0.0 IMAGE2CAD_STUDIO_PORT=8501 python -m studio.app
```

Only bind to `0.0.0.0` on a trusted network. The current service has no authentication because it is intended for local research use.

## Backend routes

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Dataset availability and the narrow write scope |
| `GET /api/summary` | Dataset and phase-level metrics |
| `GET /api/phase1/review` | Current Ring01 review manifest |
| `GET /api/phase1/assets/<view>/<kind>` | Whitelisted source, proposal or review image |
| `POST /api/phase1/reviews/label` | Record one named mask-label decision |
| `POST /api/phase1/reviews/identity` | Record one named component-identity decision |
| `POST /api/phase1/reviews/camera` | Record view semantics and camera uncertainty |
| `GET /api/phase1/stl1/review` | Dataset-wide review gate and per-object progress |
| `GET /api/phase1/stl1/review/<object_id>` | One ring's complete Phase 1 review record |
| `GET /api/phase1/stl1/assets/<object_id>/<view>/<evidence_type>` | Whitelisted proposal, correction or active evidence |
| `POST /api/phase1/stl1/reviews/evidence` | Record one named evidence decision |
| `POST /api/phase1/stl1/reviews/view` | Record one named decision across the seven inspected items in one view |
| `POST /api/phase1/stl1/reviews/object` | Record one cross-view component inventory |
| `POST /api/phase1/stl1/corrections/silhouette` | Store a separate corrected binary mask |
| `GET /api/phase4/stl1-candidates` | Dataset-wide candidate semantic-review gate |
| `GET /api/phase4/stl1-candidates/<object_id>` | Candidate proposals and decisions for one ring |
| `GET /api/phase4/stl1-candidates/<object_id>/<view>/<kind>` | Whitelisted source, overlay or candidate mask |
| `POST /api/phase4/stl1-candidates/review` | Record one identified image-space semantic decision |
| `GET /api/objects?split=test` | Object index with optional split filter |
| `GET /api/objects/<object_id>` | Five-view metadata and phase evidence |
| `GET /api/objects/<object_id>/assets/<kind>` | Whitelisted image or 3D artifact |
| `GET /api/objects/<object_id>/report` | Downloadable evidence JSON |

Asset paths are resolved against the repository root and only named artifact kinds are accepted. The service does not expose arbitrary filesystem paths. Review writes are restricted to the Ring01, STL-1 Phase 1 and STL-1 Phase 4 candidate manifests plus `phase1_human_review_v1/corrections/silhouette`; every write requires a reviewer name.

## Validate

The service layer has no third-party dependency:

```bash
python -m unittest tests.test_dataset_studio -v
```

When Flask is installed, the same test module also checks the live API contract using Flask's in-process test client.
