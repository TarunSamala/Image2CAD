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
  `evaluate_vggt_negative_space.py`, and
  `evaluate_vggt_source_camera_alignment.py`, and
  `evaluate_vggt_rigid_source_registration.py`,
  `extract_vggt_track_evidence.py`, and
  `evaluate_vggt_track_guided_registration.py`, and
  `evaluate_vggt_cycle_consistent_registration.py`, and
  `evaluate_vggt_projection_noise.py` - Bible Phase 3 research benchmark,
  low-VRAM execution, source-view transfer, camera-space silhouette completion,
  source-surface constrained hull validation, and signed-depth free-space
  carving plus rejected normal, aperture, image-plane camera and rigid
  point-map registration experiments. The VGGT track experiment is also
  rejected because its validation gain did not transfer to the fixed test set.
  Increasing coverage from 96 to 384 queries per view and requiring reciprocal
  cycles also failed: validation selected identity, so the denser tracks are
  retained as future component-relation evidence rather than geometry motion.
  Fragment-gated projection cleanup is retained for review rendering only; it
  does not alter or validate the signed-depth voxel geometry.
- `bible_phase_2_paired_benchmark.py` - Bible Phase 2 guided-capture,
  production-STEP, measurement, component-graph, split and provenance guardrail;
  inactive under the image-only execution profile.
- `image_only_reflection_correction.py` - active Ring01 reflection-sensitive
  inner-boundary correction with an explicit uncertainty mask.
- Phase 0 is primarily Docker, dependency, manifest and regression evidence.
- `build_phase4_prong_instances.py` - review-gated Ring01 instance masks for
  front/top/angled views plus explicit side/back projection groups.
- `build_phase4_stl1_component_candidates.py` - dataset-wide enclosed aperture
  and openwork masks without fabricated gemstone semantics.
- `phase4_candidate_review.py` - immutable-proposal, append-only semantic review
  records that never imply cross-view identity or CAD authorization.
- `evaluate_phase4_component_tracks.py` - non-metric VGGT point-map descriptors
  and exhaustive identity-permutation checks for the resolved Ring01 prongs.
- `component_constraint_graph.py` and `build_phase4_component_graph.py` - Bible
  Phase 4 stable-ID graph schema, provenance/uncertainty contract, directed
  edit dependencies and the synthetic ten-stone graph-state locality benchmark.
  Ring01 remains review-gated. `benchmark_phase4_exact_cad_locality.py` maps
  the synthetic graph to separate CadQuery/OCP stones, seats, cavities and
  prongs, then verifies exact B-rep locality and Boolean health for 30 edits.
  This validates designed synthetic truth only, not image-derived CAD. Run the
  prong-instance and track evaluators before rebuilding active v3 graph artifacts.
- `exact_cad_v2.py` and `benchmark_phase5_exact_cad_v2.py` - Bible Phase 5
  designed-truth structural generators and four-family exact B-rep/STEP
  benchmark. These validate the CAD layer only; image-derived fitting and
  manufacturing claims remain blocked.
- `propose_phase5_stl1_ring_parameters.py` - ring-only STL-1 bridge that
  measures all 120 silhouettes, detects each top-view shank aperture and emits
  normalized parameter proposals for all 24 rings without fabricating physical
  units or authorizing CAD.
- `fit_phase5_stl1_ring_bodies.py` - Phase 5.1 periodic-spline exact shank
  fitting for all 24 STL-1 rings. It exports valid STEP bodies, scores train and
  validation top-view fits, and keeps all three test objects unscored.
- `fit_phase5_stl1_head_supports.py` - Phase 5.2 two-view visual-hull fitting.
  It intersects front and top silhouette prisms, joins exact shanks and shoulder
  supports, exports one connected STEP candidate per ring, and retains actual
  CAD previews for the validation split. Relief and semantic detail remain open.
- `fit_phase5_stl1_openwork.py` - Phase 5.3 exact negative-space cut trials.
  It reuses all 168 stable Phase 4 front candidates, records Boolean acceptance
  per ID, exports connected STEP candidates and keeps semantic/material claims
  unresolved. This is open-space geometry evidence, not relief reconstruction.
- `build_phase5_stl1_relief_evidence.py` - Phase 5.4 conservative relief
  evidence across all 120 STL-1 views. It combines depth residuals, image-detail
  support and uncertainty, writes rejected photometric detail separately, and
  exports one reversible front OBJ layer per ring without modifying exact STEP.
- `evaluate_phase5_relief_tracks.py` - Phase 5.5/5.6 sparse reciprocal relief
  consistency. The default preserves the seven-ring checkpoint; `--all-stl1`
  applies the same frozen gate to all 24 rings. It requires source and target
  relief support plus bounded forward/reverse cycles, retains test rings as
  non-aggregated evidence, and never moves or authorizes CAD geometry.
- `extract_vggt_track_evidence.py --query-selection relief_stratified` -
  Phase 5.7 query-density research. It allocates queries across depth-supported
  relief, silhouette boundaries, textured metal and a deterministic surface
  fill. `--train-only` protects validation and test while the policy is tuned.
- `compose_vggt_hybrid_evidence.py` - preserves all 384 Phase 5.6 queries and
  appends 640 unique relief-stratified queries from the same pinned VGGT
  checkpoint. The composition is exact evidence reuse, not a new geometry run.
- `evaluate_phase5_dense_relief_pilot.py` - compares baseline, stratified-only
  and hybrid evidence with the frozen reciprocal gate. Baseline acceptance is
  retained exactly; appended tracks can add review evidence but cannot move or
  authorize CAD.
- `evaluate_phase5_normal_corroboration.py` - Phase 5.8 validation annotation.
  It checks frozen hybrid tracks against Metric3D direct normals and a separate
  Depth Anything V2 plus UniDepth normal consensus. Metric3D depth is excluded
  from the comparator; unresolved tracks stay visible and CAD remains unchanged.
- `build_phase5_visibility_graph.py` - Phase 5.9 validation-only observability
  graph. It retains every frozen reciprocal edge, records which named target
  views actually observe each front query, annotates normal support, and groups
  nearby supported points only when they share a target view. The graph routes
  evidence for review; it does not infer hidden surfaces or modify CAD.
- `build_phase5_component_evidence_graph.py` - Phase 5.10 validation-only
  component bridge. It assigns stable IDs to visibility regions, parents them
  to the exact metal/head candidates, records deterministic spatial zones and
  links nearby Phase 4 negative-space evidence. Semantic labels stay unknown;
  no stone, prong, face, cavity or CAD-fusion claim is created.
- `subdivide_phase5_component_evidence.py` - Phase 5.11 validation-only relief
  subdivision. It splits broad components using bounded spatial links, shared
  reciprocal target views, regularized relative-depth continuity, prepared-edge
  barriers and a maximum patch diameter. Singleton evidence remains unresolved;
  patches are not semantic labels or CAD surfaces.
- `build_phase5_local_relief_surfaces.py` - Phase 5.12 reversible local surface
  trial. It triangulates only the 40 validation patches already eligible for
  fitting review, samples conservative signed-relief evidence and writes open
  nominal-scale OBJ groups with same-view interpolation audits. It does not
  process test rings, infer semantics, generate STEP or authorize CAD fusion.
- `validate_phase5_local_relief_reprojection.py` - Phase 5.13 independent
  target-view gate. It projects unchanged Phase 5.12 triangle connectivity
  through normal-corroborated reciprocal tracks and compares source-derived
  relief with target-view evidence without refitting. Weak and unresolved
  results remain visible; no surface, semantic or CAD claim is promoted.
- `diagnose_phase5_target_reprojection.py` - Phase 5.14 frozen-gate failure
  attribution. It reproduces the Phase 5.13 counts, classifies structural and
  relief-transfer blockers, and writes a ranked training-split action queue.
  It does not change thresholds, surfaces, validation scores or CAD.
- `evaluate_phase5_training_track_topology.py` - Phase 5.15 training-only
  reciprocal topology experiment. It compares the Phase 5.6 baseline with the
  Phase 5.7 hybrid query policy, forms bounded local triangles only from three
  same-target accepted vertices, and measures non-degenerate,
  orientation-preserved projections. It does not process validation or test
  rings, fit relief, create surfaces or authorize CAD.
- `evaluate_phase5_validation_track_topology.py` - Phase 5.16 frozen-policy
  transfer to the three validation rings. It verifies the Phase 5.15 integrity
  manifest before using the same topology construction and reports each target
  view separately. It does not update the Phase 5.13 relief-surface gate.
- `evaluate_phase5_training_relief_descriptors.py` - Phase 5.17 training-only
  comparison of signed relief with unsigned depth detail, photo detail and a
  fused descriptor. Accepted reciprocal tracks are compared with displaced
  controls matched for target relief confidence and photo-detail strength.
  None of the alternatives passes the predeclared selection gate.

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
