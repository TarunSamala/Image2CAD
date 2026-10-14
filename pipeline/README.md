# Pipeline module map

The Project Bible v1.0 Phase 0-9 roadmap is canonical. Many module filenames
predate it, so this index prevents similarly numbered stages from being mixed.

## Bible implementation modules

- `phase1_ground_truth.py`, `phase1_review.py`, `stl1_phase1_review.py` - Bible
  Phase 1 evidence and identified human review.
- `run_phase3_geometry_benchmark.py`, `vggt_geometry_adapter.py`,
  `run_vggt_multiring_experiment.py`, `validate_vggt_multiring_experiment.py`,
  `compare_vggt_resolution_ladder.py`, `evaluate_vggt_view_transfer.py`, and
  `evaluate_vggt_visual_hull_completion.py`,
  `evaluate_vggt_surface_aware_hull.py`, and
  `evaluate_vggt_signed_depth_carving.py`,
  `evaluate_vggt_normal_consistency.py`, and
  `evaluate_vggt_negative_space.py` - Bible Phase 3 research benchmark,
  low-VRAM execution, source-view transfer, camera-space silhouette completion,
  source-surface constrained hull validation, and signed-depth free-space
  carving plus rejected normal/aperture constraint experiments.
- `bible_phase_2_paired_benchmark.py` - Bible Phase 2 guided-capture,
  production-STEP, measurement, component-graph, split and provenance guardrail;
  inactive under the image-only execution profile.
- `image_only_reflection_correction.py` - active Ring01 reflection-sensitive
  inner-boundary correction with an explicit uncertainty mask.
- Phase 0 is primarily Docker, dependency, manifest and regression evidence.

## Shared evidence and dataset tools

- `prepare_jewellery_dataset.py`, `jewellery_multiview_dataset.py`
- `extract_dataset_complete_phase1.py`, `infer_dataset_depth_anything_v2.py`
- `audit_jewellery.py`, comparison and audit builders
- `intake_external_ring_datasets.py` - provenance-safe OrnAsia and Segment Your
  Ring intake with strict human-mask gates; unlabelled masks remain
  review-required.
- `bible_image_only_phase2_machine_review.py` - deterministic and AI-assisted
  audit that permits image-only Phase 2 research without modifying or
  impersonating the human Phase 1 review gate.
- metadata, schema, validation-sheet and camera/depth support modules

`audit_jewellery.py` includes an experimental compact-object SAM prompt for
small jewellery on non-uniform backgrounds. It rejects long background folds,
records the prompt strategy and candidate diagnostics, and adds a negative
centre prompt when local colour evidence indicates a central opening. These
heuristics improve representative real-image rings but do not replace human
mask validation.

Some shared filenames contain old phase labels because their generated
artifacts are immutable legacy evidence.

## Legacy Ring01 and dataset modules

Modules named `extract_phase1`, `extract_phase2*`, `refine_phase2*`,
`reconstruct_phase3`, `build_phase3*`, `refine_phase3*`, `validate_phase3*`,
`train_dataset_phases` and `reconstruct_dataset_phase3` implement the pre-Bible
pipeline. Keep them for regression and backtracking; do not infer Bible phase
completion from their names.

## Naming new work

Use `bible_phase_<number>_<capability>.py` when a new module is specific to a
Bible phase, or use a clear capability name when it is shared. Do not add a new
ambiguous bare `phase1`, `phase2` or `phase3` module.
