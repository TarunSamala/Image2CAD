# Bible implementation status

This directory contains implementation evidence mapped to the canonical
[Project Bible v1.0 roadmap](../bible_v1_0/19_DEVELOPMENT_ROADMAP.md).

Active execution profile: **images only**. Production CAD, scans, physical
measurements and calibrated cameras are unavailable as inputs. The frozen Bible
remains unchanged, while implementation reports explicitly mark gates that
cannot apply under this constraint.

| Bible phase | Status | Evidence |
| --- | --- | --- |
| Phase 0 - Reproducibility Freeze | Complete | [Status and manifests](phase_0_reproducibility/PHASE0_STATUS.md) |
| Phase 1 - Ground-Truth Ring01 | In progress | [Image evidence](phase_1_ground_truth/IMAGE_ONLY_GROUND_TRUTH.md), [STL-1 review gate](phase_1_ground_truth/STL1_HUMAN_REVIEW.md) |
| Phase 2 - Real Paired Benchmark | Not applicable to the image-only profile | [Retained guardrail](phase_2_paired_benchmark/README.md); not an active collection request |
| Image-only Phase 2 - Reflection and metric-depth evidence | Machine-reviewed research may continue; human and metric-scale gates blocked | [Machine-review checkpoint](../../data/ring01_image_only/machine_review_v1/MACHINE_REVIEW.md), [metric-depth consensus](image_only_phase_2/METRIC_DEPTH_CONSENSUS.md), [Ring01 correction report](../../data/ring01_image_only/reflection_correction_v1/report.json) |
| Phase 3 - Camera / Geometry Evidence Benchmark | Partial research implementation | [Benchmark contract](phase_3_geometry_benchmark/README.md) |
| Phase 4-9 | Not started | Defined in the canonical roadmap |

Phase 1 machine proposals are ready, but identified human review, one known
physical dimension for Ring01 and the remaining ground-truth decisions are not
complete. Existing legacy Phase 3.3 Ring01 CAD does not imply completion of
Bible Phase 3.

The image-only Phase 2 machine-review checkpoint verifies retained hashes,
binary masks, all 840 STL-1 evidence references and the Ring01 reflection
correction. It records AI-assisted visual findings separately and does not write
to either human-review manifest. This permits research continuation with
uncertainty; it does not complete Bible Phase 1 or canonical paired-CAD Phase 2.

[Lalitha Studio](STUDIO.md) is the review interface for current evidence.

The current implementation position and remaining work are summarized in the
[progress roadmap](PROGRESS_ROADMAP.md).
