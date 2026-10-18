"""Regression tests for the read-only STL-1 dataset backend."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from studio.dataset_service import (
    AssetNotFoundError,
    DatasetRepository,
    ObjectNotFoundError,
    VIEW_ORDER,
)


class DatasetStudioServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = DatasetRepository(Path.cwd())

    def test_summary_uses_all_twenty_four_five_view_objects(self) -> None:
        summary = self.repository.summary()
        self.assertEqual(summary["dataset"]["object_count"], 24)
        self.assertEqual(summary["dataset"]["image_count"], 120)
        self.assertEqual(summary["dataset"]["views"], list(VIEW_ORDER))
        self.assertTrue(summary["dataset"]["valid"])
        self.assertEqual(summary["governance"]["capture_tier"], "A")
        self.assertEqual(summary["governance"]["decision"], "research_only")
        self.assertEqual(
            summary["governance"]["authoritative_output"], "STEP / exact B-rep"
        )
        self.assertRegex(summary["phase0"]["source"]["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(summary["phase0"]["regression"]["result"], "passed")
        self.assertGreaterEqual(summary["phase0"]["dependency_package_count"], 20)
        self.assertEqual(summary["phase1"]["view_count"], 120)
        self.assertFalse(summary["phase1"]["measurements_are_metric"])
        dataset_depth = summary["phase1"]["depth_anything_v2"]
        self.assertEqual(dataset_depth["coverage"]["object_count"], 24)
        self.assertEqual(dataset_depth["coverage"]["view_count"], 120)
        self.assertTrue(all(dataset_depth["checks"].values()))
        self.assertEqual(dataset_depth["phase1_gate"]["status"], "blocked")
        self.assertFalse(dataset_depth["phase1_gate"]["metric_depth"])
        complete = summary["phase1"]["complete_features"]
        self.assertEqual(complete["coverage"]["object_count"], 24)
        self.assertEqual(complete["coverage"]["view_count"], 120)
        self.assertEqual(complete["runtime"]["repeat_count"], 3)
        self.assertTrue(all(complete["checks"].values()))
        self.assertEqual(
            complete["feature_contract"]["stored_array_count_per_view"], 22
        )
        self.assertTrue(
            complete["reconstruction_contract"]["source_pixel_roundtrip_exact"]
        )
        self.assertEqual(complete["metric_depth"]["status"], "blocked")
        ground_truth = summary["phase1_ground_truth"]
        self.assertEqual(ground_truth["exit_gate"]["status"], "blocked")
        self.assertEqual(ground_truth["exit_gate"]["pending_label_review_count"], 35)
        self.assertEqual(
            ground_truth["depth"]["status"], "machine_proposals_pending_review"
        )
        self.assertFalse(ground_truth["depth"]["metric"])
        self.assertTrue(
            ground_truth["depth_validation"]["checks"][
                "all_three_repeats_identical"
            ]
        )
        self.assertEqual(
            ground_truth["depth_validation"]["metric_depth"]["status"],
            "blocked",
        )
        self.assertEqual(ground_truth["pixel_features"]["view_count"], 5)
        self.assertTrue(
            ground_truth["pixel_features"]["reconstruction_contract"][
                "pixel_exact_roundtrip"
            ]
        )
        self.assertFalse(
            ground_truth["pixel_features"]["depth_contract"][
                "metric_depth_available"
            ]
        )
        stl1_review = summary["stl1_phase1_review"]
        self.assertEqual(stl1_review["gate"]["object_count"], 24)
        self.assertEqual(stl1_review["gate"]["view_count"], 120)
        self.assertEqual(stl1_review["gate"]["evidence_review_count"], 840)
        self.assertEqual(stl1_review["gate"]["pending_evidence_count"], 840)
        self.assertEqual(stl1_review["gate"]["pending_object_review_count"], 24)
        external = summary["external_ring_datasets"]
        self.assertEqual(external["status"], "review_required")
        self.assertEqual(external["sample_count"], 206)
        self.assertEqual(external["dataset_counts"]["ornasia"], 196)
        self.assertEqual(external["ground_truth_mask_count"], 10)
        self.assertEqual(external["approved_mask_count"], 1)
        self.assertEqual(external["review_required_mask_count"], 205)
        self.assertFalse(external["checks"]["all_segmentation_outputs_approved"])
        phase2 = summary["phase2"]
        self.assertEqual(
            phase2["status"], "correction_proposed_pending_human_review"
        )
        self.assertEqual(phase2["input_contract"], "images_only")
        self.assertTrue(phase2["checkpoint_valid"])
        machine_review = phase2["machine_review"]
        self.assertTrue(machine_review["claims"]["machine_evidence_validated"])
        self.assertTrue(
            machine_review["claims"]["image_only_phase2_research_may_continue"]
        )
        self.assertFalse(machine_review["claims"]["human_review_performed"])
        self.assertFalse(machine_review["claims"]["bible_phase1_complete"])
        metric_depth = phase2["metric_depth_consensus"]
        self.assertGreaterEqual(len(metric_depth["available_backends"]), 3)
        self.assertEqual(metric_depth["sample_count"], 20)
        self.assertEqual(
            metric_depth["incomplete_backends"]["depth_pro"]["missing_sample_count"],
            16,
        )
        self.assertTrue(metric_depth["claims"]["cross_model_stability_measured"])
        self.assertFalse(metric_depth["claims"]["depth_accuracy_measured"])
        self.assertEqual(
            metric_depth["visualization"]["depth_color_legend"],
            {"red": "near", "blue": "far"},
        )
        self.assertEqual(
            metric_depth["visualization"]["disagreement_color_legend"],
            {"red": "high_disagreement", "blue": "low_disagreement"},
        )
        self.assertFalse(
            metric_depth["geometry_alignment"]["cross_view_metric_alignment_complete"]
        )
        normals = phase2["surface_normal_evidence"]
        self.assertEqual(normals["sample_count"], 20)
        self.assertTrue(normals["claims"]["per_pixel_normal_hypotheses_available"])
        self.assertFalse(normals["claims"]["normal_accuracy_measured"])
        self.assertEqual(
            normals["backends"]["dsine"]["status"],
            "blocked_license_public_deployment",
        )
        readiness = summary["phase3"]["geometry_benchmark"]["cross_view_readiness"]
        self.assertEqual(readiness["summary"]["candidate_count"], 4)
        self.assertEqual(readiness["summary"]["public_product_candidates"], ["vggt"])
        self.assertEqual(readiness["summary"]["locally_ready_count"], 0)
        vggt_experiment = summary["phase3"]["geometry_benchmark"][
            "ring01_vggt_low_vram"
        ]
        self.assertTrue(vggt_experiment["checks"]["run_completed"])
        self.assertEqual(vggt_experiment["experiment"]["image_count"], 5)
        self.assertEqual(
            vggt_experiment["machine_visual_review"]["decision"],
            "rejected_as_cross_view_reconstruction_geometry",
        )
        multiring_vggt = summary["phase3"]["geometry_benchmark"][
            "vggt_multiring_low_vram"
        ]
        self.assertEqual(multiring_vggt["object_count"], 3)
        self.assertEqual(multiring_vggt["accepted_object_count"], 0)
        self.assertEqual(
            [item["object_id"] for item in multiring_vggt["objects"]],
            ["ring_001", "ring_007", "ring_023"],
        )
        multiring_refinement = summary["phase3"]["geometry_benchmark"][
            "vggt_multiring_refinement"
        ]
        self.assertLess(multiring_refinement["cohort"]["delta"], 0)
        self.assertEqual(
            multiring_refinement["selected_configuration"],
            "prepared_v1_full_frame",
        )
        semantic_priors = summary["phase3"]["geometry_benchmark"][
            "semantic_camera_priors"
        ]
        self.assertEqual(semantic_priors["object_count"], 3)
        self.assertLess(
            semantic_priors["cohort"]["mean_vggt_pairwise_angle_error_deg"],
            15.0,
        )
        semantic_refinement = summary["phase3"]["geometry_benchmark"][
            "semantic_camera_refinement"
        ]
        self.assertLess(semantic_refinement["cohort"]["mean_iou_delta"], 0)
        self.assertEqual(
            semantic_refinement["cohort"]["accepted_initialization_count"], 0
        )
        resolution = summary["phase3"]["geometry_benchmark"][
            "vggt_resolution_ladder"
        ]
        self.assertEqual(resolution["selected_resolution"], 518)
        self.assertGreater(resolution["heldout_change_from_224"]["mean_iou"], 0.3)
        self.assertFalse(
            resolution["checks"]["all_selected_objects_pass_geometry_gate"]
        )
        transfer = summary["phase3"]["geometry_benchmark"][
            "vggt_view_transfer"
        ]
        self.assertLess(transfer["heldout"]["mean_iou_delta"], 0)
        signed_depth = summary["phase3"]["geometry_benchmark"][
            "vggt_signed_depth_carving"
        ]
        self.assertGreater(signed_depth["test_evaluation"]["mean_iou"], 0.78)
        self.assertFalse(signed_depth["checks"]["geometry_exit_gate_passed"])
        normal = summary["phase3"]["geometry_benchmark"][
            "vggt_normal_consistency"
        ]
        self.assertEqual(normal["validation_selection"]["selected_normal_cosine_threshold"], 0.0)
        negative_space = summary["phase3"]["geometry_benchmark"][
            "vggt_negative_space"
        ]
        self.assertEqual(
            negative_space["validation_selection"]["selected_hole_margin_pixels"],
            4,
        )
        camera_alignment = summary["phase3"]["geometry_benchmark"][
            "vggt_source_camera_alignment"
        ]
        self.assertEqual(
            camera_alignment["validation_selection"]["selected_profile"],
            "identity",
        )
        self.assertEqual(
            camera_alignment["decision"],
            "rejected_no_consistent_heldout_gain",
        )
        rigid_registration = summary["phase3"]["geometry_benchmark"][
            "vggt_rigid_source_registration"
        ]
        self.assertEqual(
            rigid_registration["validation_selection"]["selected_profile"],
            "identity",
        )
        self.assertEqual(
            rigid_registration["decision"],
            "rejected_no_consistent_heldout_gain",
        )
        track_evidence = summary["phase3"]["geometry_benchmark"][
            "vggt_track_evidence"
        ]
        self.assertEqual(track_evidence["object_count"], 7)
        self.assertTrue(track_evidence["checks"]["all_objects_completed"])
        track_registration = summary["phase3"]["geometry_benchmark"][
            "vggt_track_guided_registration"
        ]
        self.assertEqual(
            track_registration["validation_selection"]["selected_profile"],
            "track_strict",
        )
        self.assertEqual(
            track_registration["decision"],
            "rejected_no_consistent_heldout_gain",
        )
        dense_tracks = summary["phase3"]["geometry_benchmark"][
            "vggt_dense_track_evidence"
        ]
        self.assertEqual(dense_tracks["object_count"], 7)
        self.assertEqual(dense_tracks["objects"][0]["query_count_per_view"], 384)
        dense_cycle = summary["phase3"]["geometry_benchmark"][
            "vggt_dense_cycle_registration"
        ]
        self.assertEqual(
            dense_cycle["validation_selection"]["selected_profile"],
            "identity",
        )
        self.assertEqual(
            dense_cycle["decision"],
            "rejected_no_regression_free_heldout_gain",
        )
        projection_noise = summary["phase3"]["geometry_benchmark"][
            "vggt_projection_noise"
        ]
        self.assertEqual(
            projection_noise["validation_selection"]["selected_profile"],
            "close3_components_ge_4",
        )
        self.assertEqual(
            projection_noise["decision"],
            "retained_as_projection_renderer_only",
        )
        self.assertEqual(
            phase2["ring01_carryover"]["open_issues"][0]["status"],
            "correction_proposed_pending_human_review",
        )
        self.assertEqual(
            summary["phase2_paired"]["applicability"],
            "not_applicable_under_image_only_constraint",
        )
        self.assertIn("test", summary["legacy_phase2"])
        geometry = summary["phase3"]["geometry_benchmark"]
        self.assertEqual(geometry["selected_object_count"], 24)
        self.assertEqual(geometry["backends"]["colmap"]["version"], "4.2.1")
        self.assertEqual(geometry["backends"]["vggt"]["status"], "blocked")
        phase4 = summary["phase4"]
        self.assertEqual(phase4["ring01"]["node_count"], 10)
        self.assertEqual(
            phase4["synthetic_ten_stone_edit_benchmark"]["passed_case_count"],
            30,
        )
        self.assertEqual(
            phase4["exact_cad_locality"]["summary"]["passed_case_count"], 30
        )
        self.assertTrue(
            phase4["exact_cad_locality"]["checks"][
                "all_unrelated_exact_breps_unchanged"
            ]
        )
        self.assertEqual(summary["phase5"]["summary"]["passed_family_count"], 4)
        self.assertFalse(
            summary["phase5"]["claims"]["image_derived_exact_cad_validated"]
        )
        self.assertEqual(
            summary["phase5"]["stl1_ring_parameters"]["summary"][
                "parameterized_ring_count"
            ],
            24,
        )
        self.assertTrue(
            summary["phase5"]["stl1_ring_parameters"]["checks"][
                "ring_only_scope_enforced"
            ]
        )
        self.assertEqual(
            summary["phase5"]["stl1_ring_body_fit"]["summary"]["object_count"],
            24,
        )
        self.assertTrue(
            summary["phase5"]["stl1_ring_body_fit"]["checks"][
                "held_out_test_not_scored"
            ]
        )
        self.assertEqual(
            summary["phase5"]["stl1_head_support_fit"]["summary"][
                "object_count"
            ],
            24,
        )
        self.assertTrue(
            summary["phase5"]["stl1_head_support_fit"]["checks"][
                "all_metals_valid_single_solids"
            ]
        )
        self.assertEqual(
            summary["phase5"]["stl1_openwork_fit"]["summary"][
                "input_candidate_count"
            ],
            168,
        )
        self.assertTrue(
            summary["phase5"]["stl1_openwork_fit"]["checks"][
                "candidate_semantics_not_promoted"
            ]
        )
        self.assertEqual(
            summary["phase5"]["stl1_relief_evidence"]["summary"][
                "view_count"
            ],
            120,
        )
        self.assertTrue(
            summary["phase5"]["stl1_relief_evidence"]["checks"][
                "relief_kept_separate_from_exact_cad"
            ]
        )
        self.assertFalse(
            summary["phase5"]["stl1_relief_evidence"]["claims"][
                "relief_accuracy_measured"
            ]
        )
        self.assertEqual(
            summary["phase5"]["relief_track_consistency"]["summary"][
                "object_count"
            ],
            7,
        )
        self.assertTrue(
            summary["phase5"]["relief_track_consistency"]["checks"][
                "reciprocal_cycle_required"
            ]
        )
        self.assertFalse(
            summary["phase5"]["relief_track_consistency"]["claims"][
                "cad_fusion_authorized"
            ]
        )
        self.assertEqual(
            summary["phase5"]["full_relief_track_consistency"]["summary"][
                "object_count"
            ],
            24,
        )
        self.assertEqual(
            summary["phase5"]["full_relief_track_consistency"]["summary"][
                "view_count"
            ],
            120,
        )
        self.assertTrue(
            summary["phase5"]["full_relief_track_consistency"]["checks"][
                "held_out_test_not_aggregated"
            ]
        )
        self.assertFalse(
            summary["phase5"]["full_relief_track_consistency"]["claims"][
                "cad_fusion_authorized"
            ]
        )
        dense_train = summary["phase5"]["dense_relief_training"]
        self.assertEqual(dense_train["object_count"], 18)
        self.assertTrue(dense_train["checks"]["baseline_acceptance_gate_preserved"])
        dense_validation = summary["phase5"]["dense_relief_validation"]
        self.assertEqual(dense_validation["object_count"], 3)
        self.assertTrue(
            dense_validation["summary"][
                "all_objects_improve_supported_spatial_coverage"
            ]
        )
        self.assertTrue(
            dense_validation["checks"]["baseline_acceptance_gate_preserved"]
        )
        self.assertFalse(dense_validation["claims"]["cad_fusion_authorized"])
        normal_corroboration = summary["phase5"]["normal_corroboration"]
        self.assertEqual(normal_corroboration["summary"]["object_count"], 3)
        self.assertEqual(
            normal_corroboration["summary"]["normal_corroborated_count"], 521
        )
        self.assertEqual(
            normal_corroboration["summary"]["normal_unresolved_count"], 215
        )
        self.assertTrue(
            normal_corroboration["checks"][
                "metric3d_depth_excluded_from_independent_comparator"
            ]
        )
        self.assertFalse(normal_corroboration["claims"]["cad_fusion_authorized"])
        self.assertFalse(phase4["checks"]["phase4_exit_gate_passed"])
        self.assertTrue(phase4["checks"]["partial_ring01_prong_instances_available"])
        self.assertEqual(
            phase4["prong_instances"]["summary"]["resolved_instance_view_count"],
            3,
        )
        self.assertTrue(
            phase4["component_tracks"]["gate"][
                "resolved_identity_assignments_machine_supported"
            ]
        )
        self.assertEqual(
            phase4["component_tracks"]["comparisons"]["top"]["evidence_level"],
            "strong_machine_support",
        )
        candidates = phase4["stl1_component_candidates"]
        self.assertEqual(candidates["object_count"], 24)
        self.assertEqual(candidates["view_count"], 120)
        self.assertEqual(candidates["candidate_count"], 266)
        self.assertFalse(candidates["gate"]["individual_stones_confirmed"])
        self.assertFalse(
            candidates["gate"]["component_graph_generation_authorized"]
        )
        candidate_review = phase4["stl1_candidate_review"]
        self.assertEqual(candidate_review["gate"]["candidate_count"], 266)
        self.assertEqual(candidate_review["gate"]["pending_candidate_count"], 266)
        self.assertFalse(
            candidate_review["gate"]["component_graph_generation_authorized"]
        )

    def test_object_filters_preserve_object_level_splits(self) -> None:
        self.assertEqual(len(self.repository.list_objects("train")), 18)
        self.assertEqual(len(self.repository.list_objects("val")), 3)
        self.assertEqual(len(self.repository.list_objects("test")), 3)
        self.assertEqual(len(self.repository.list_objects("all")), 24)

    def test_capability_console_reports_working_and_open_functionality(self) -> None:
        payload = self.repository.capabilities()
        self.assertEqual(payload["schema_version"], "studio_capabilities_v1")
        self.assertEqual(payload["execution_profile"], "images_only")
        self.assertEqual(payload["summary"]["capability_count"], 10)
        self.assertEqual(payload["summary"]["working_count"], 4)
        self.assertEqual(payload["summary"]["partial_count"], 6)
        self.assertEqual(payload["summary"]["blocked_count"], 0)
        indexed = {item["id"]: item for item in payload["capabilities"]}
        self.assertEqual(indexed["feature_extraction"]["status"], "implemented")
        self.assertIn("120 views", indexed["feature_extraction"]["coverage"])
        self.assertTrue(
            indexed["feature_extraction"]["checks"]["source_pixel_roundtrip_exact"]
        )
        self.assertEqual(indexed["depth_evidence"]["status"], "partial")
        self.assertIn("4/20", indexed["depth_evidence"]["limitations"][0])
        self.assertEqual(indexed["component_graph"]["status"], "partial")
        self.assertTrue(
            indexed["component_graph"]["checks"][
                "synthetic_ten_stone_edit_test_passed"
            ]
        )
        self.assertEqual(indexed["parametric_cad"]["status"], "partial")
        self.assertEqual(len(payload["remaining_gates"]), 4)

    def test_every_object_exposes_real_view_and_phase_artifacts(self) -> None:
        for item in self.repository.list_objects():
            detail = self.repository.object_detail(item["object_id"])
            self.assertEqual(item["phase1_pending_review_count"], 36)
            self.assertEqual(detail["phase1_review"]["gate"]["status"], "blocked")
            self.assertEqual(detail["view_order"], list(VIEW_ORDER))
            self.assertEqual(detail["bible"]["capture"]["tier"], "A")
            self.assertEqual(detail["bible"]["decision"], "research_only")
            self.assertFalse(any(detail["bible"]["gates"].values()))
            self.assertTrue(detail["artifacts"]["phase_sheet"])
            self.assertTrue(detail["artifacts"]["phase1_depth_review"])
            self.assertTrue(detail["artifacts"]["phase1_complete_audit"])
            self.assertTrue(detail["artifacts"]["phase3_preview"])
            self.assertTrue(detail["artifacts"]["phase3_stl"])
            self.assertTrue(detail["artifacts"]["phase3_3mf"])
            self.assertTrue(detail["artifacts"]["colmap_report"])
            self.assertEqual(
                detail["geometry_evidence"]["authority"], "machine_hypothesis"
            )
            for view in VIEW_ORDER:
                self.assertTrue(all(detail["artifacts"]["views"][view].values()))
                self.assertEqual(
                    set(detail["views"][view]["features"]),
                    {
                        "foreground_fraction",
                        "edge_fraction",
                        "contour_count",
                        "hole_count",
                        "horizontal_symmetry_iou",
                    },
                )
                self.assertFalse(detail["views"][view]["depth"]["metric"])
                self.assertFalse(
                    detail["views"][view]["depth"]["cross_view_aligned"]
                )
                complete = detail["views"][view]["complete_features"]
                self.assertEqual(complete["feature_count"], 22)
                self.assertEqual(complete["repeat_count"], 3)
                self.assertTrue(complete["repeat_deterministic"])
                self.assertTrue(complete["source_pixel_roundtrip_exact"])
                self.assertFalse(complete["metric"])
                self.assertFalse(complete["cross_view_aligned"])

    def test_test_objects_expose_held_out_review_sheets(self) -> None:
        for item in self.repository.list_objects("test"):
            self.assertTrue(
                self.repository.object_detail(item["object_id"])["artifacts"][
                    "held_out_review"
                ]
            )

    def test_export_report_keeps_accuracy_claims_honest(self) -> None:
        report = self.repository.export_object_report("ring_001")
        self.assertTrue(report["claims"]["phase2_pseudo_label_self_consistency_measured"])
        self.assertTrue(report["claims"]["phase1_relative_depth_proposals_generated"])
        self.assertFalse(report["claims"]["phase1_depth_accuracy_validated"])
        self.assertFalse(report["claims"]["phase2_human_ground_truth_accuracy_validated"])
        self.assertTrue(report["claims"]["phase3_experimental_non_metric"])
        self.assertFalse(report["claims"]["manufacturing_accuracy_validated"])
        self.assertTrue(
            report["benchmark_record"]["code_commit_recorded_in_source_report"]
        )
        self.assertTrue(report["benchmark_record"]["license_manifest_recorded"])
        self.assertTrue(
            report["benchmark_record"]["model_checkpoint_revision_recorded"]
        )
        self.assertTrue(report["benchmark_record"]["peak_memory_recorded"])
        self.assertTrue(report["benchmark_record"]["runtime_recorded"])
        self.assertEqual(report["benchmark_record"]["completeness"], "partial")
        self.assertEqual(report["benchmark_record"]["decision"], "research_only")
        json.dumps(report)

    def test_unknown_object_and_asset_are_rejected(self) -> None:
        with self.assertRaises(ObjectNotFoundError):
            self.repository.object_detail("ring_999")
        with self.assertRaises(AssetNotFoundError):
            self.repository.asset_path("ring_001", "../../etc/passwd")
        with self.assertRaises(AssetNotFoundError):
            self.repository.asset_path("ring_001", "source", "../../front")


try:
    import flask  # noqa: F401
except ImportError:
    flask = None


@unittest.skipIf(flask is None, "Flask is an optional Studio dependency")
class DatasetStudioApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from studio.app import create_app

        app = create_app(Path.cwd())
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_health_and_summary_routes(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["objects"], 24)
        self.assertEqual(
            health.get_json()["write_scope"],
            "identified_phase1_and_phase4_image_space_review_decisions_plus_corrected_silhouettes",
        )
        summary = self.client.get("/api/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.get_json()["dataset"]["image_count"], 120)
        capabilities = self.client.get("/api/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertEqual(
            capabilities.get_json()["summary"]["capability_count"], 10
        )
        capabilities.close()
        environment = self.client.get("/api/artifacts/phase0_environment")
        self.assertEqual(environment.status_code, 200)
        self.assertRegex(
            environment.get_json()["source"]["commit"], r"^[0-9a-f]{40}$"
        )
        environment.close()

    def test_studio_page_exposes_visual_and_metadata_evidence_controls(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        markup = page.get_data(as_text=True)
        for marker in (
            'id="cv-index-title"',
            'class="cv-layer-grid"',
            'id="metric-depth-color-convention"',
            'id="phase0-summary"',
            'id="phase0-limits"',
            'data-panel="phase4"',
            'id="phase4-status"',
            'id="phase4-preview-edit"',
            'id="phase1-summary"',
            'id="phase1-review-console"',
            'id="phase1-reviewer"',
            'id="phase1-proposal-preview"',
            'id="phase1-depth-status"',
            'id="phase1-deep-summary"',
            'id="metric-depth-consensus-summary"',
            'id="surface-normal-evidence"',
            'id="surface-normal-summary"',
            'id="cross-view-readiness"',
            'id="semantic-camera-status"',
            'id="vggt-resolution-status"',
            'id="vggt-visual-hull-status"',
            'id="vggt-surface-hull-status"',
            'id="vggt-signed-depth-status"',
            'id="vggt-normal-consistency-status"',
            'id="vggt-negative-space-status"',
            'id="vggt-source-camera-alignment-status"',
            'id="vggt-rigid-registration-status"',
            'id="vggt-track-registration-status"',
            'id="vggt-projection-noise-status"',
            'id="dataset-depth-summary"',
            'id="dataset-complete-summary"',
            'id="stl1-review-summary"',
            'id="external-ring-summary"',
            'id="panel-phase1-review"',
            'id="stl1-reviewer"',
            'id="stl1-review-evidence-image"',
            'id="stl1-corrected-mask"',
            'id="stl1-approve-view"',
            'id="stl1-has-relief"',
            'data-layer="normalized"',
            'data-layer="mask"',
            'data-layer="edges"',
            'data-layer="depth"',
            'data-layer="depth_uncertainty"',
            'data-layer="depth_ensemble"',
            'data-layer="depth_transform_uncertainty"',
            'data-layer="depth_overlay"',
            'id="phase1-complete-audit"',
            'id="phase2-paired-summary"',
            'id="phase2-blockers"',
            'id="ring01-carryover-note"',
            'id="phase3-geometry"',
            'id="phase3-geometry-report"',
            'id="feature-rows"',
            'id="depth-rows"',
        ):
            self.assertIn(marker, markup)
        self.assertNotIn("LIVE IMPLEMENTATION STATUS", markup)
        self.assertNotIn('id="implementation-console"', markup)
        script = self.client.get("/static/studio.js")
        self.assertEqual(script.status_code, 200)
        javascript = script.get_data(as_text=True)
        self.assertIn('$("#phase0-summary")', javascript)
        self.assertNotIn('api("/api/capabilities")', javascript)
        self.assertNotIn("renderCapabilities", javascript)
        self.assertIn('$("#metric-depth-color-convention")', javascript)
        self.assertIn('$("#feature-rows")', javascript)
        self.assertIn('$("#phase1-summary")', javascript)
        self.assertIn("loadPhase1Review()", javascript)
        self.assertIn("/api/phase1/reviews/label", javascript)
        self.assertIn('$("#phase1-deep-summary")', javascript)
        self.assertIn('$("#dataset-depth-summary")', javascript)
        self.assertIn('$("#dataset-complete-summary")', javascript)
        self.assertIn('$("#external-ring-summary")', javascript)
        self.assertIn("loadStl1ReviewSummary()", javascript)
        self.assertIn("/api/phase1/stl1/reviews/evidence", javascript)
        self.assertIn("/api/phase1/stl1/reviews/view", javascript)
        self.assertIn("/api/phase1/stl1/corrections/silhouette", javascript)
        self.assertIn('$("#phase1-complete-audit")', javascript)
        self.assertIn('$("#phase2-paired-summary")', javascript)
        self.assertIn('$("#metric-depth-consensus-summary")', javascript)
        self.assertIn('$("#surface-normal-summary")', javascript)
        self.assertIn('$("#cross-view-readiness")', javascript)
        self.assertIn('$("#ring01-vggt-status")', javascript)
        self.assertIn('$("#semantic-camera-status")', javascript)
        self.assertIn('$("#vggt-resolution-status")', javascript)
        self.assertIn('$("#vggt-visual-hull-status")', javascript)
        self.assertIn('$("#vggt-surface-hull-status")', javascript)
        self.assertIn('$("#vggt-signed-depth-status")', javascript)
        self.assertIn('$("#vggt-normal-consistency-status")', javascript)
        self.assertIn('$("#vggt-negative-space-status")', javascript)
        self.assertIn('$("#vggt-source-camera-alignment-status")', javascript)
        self.assertIn('$("#vggt-rigid-registration-status")', javascript)
        self.assertIn('$("#vggt-track-registration-status")', javascript)
        self.assertIn('$("#vggt-projection-noise-status")', javascript)
        self.assertIn("phase2.machine_review", javascript)
        self.assertIn("summary.phase1.complete_features", javascript)
        audit = self.client.get("/api/artifacts/phase2_reflection_audit")
        self.assertEqual(audit.status_code, 200)
        self.assertEqual(audit.mimetype, "image/png")
        audit.close()
        machine_review = self.client.get("/api/artifacts/phase2_machine_review")
        self.assertEqual(machine_review.status_code, 200)
        self.assertEqual(machine_review.mimetype, "application/json")
        machine_review.close()
        metric_depth = self.client.get("/api/artifacts/metric_depth_consensus_report")
        self.assertEqual(metric_depth.status_code, 200)
        self.assertEqual(metric_depth.mimetype, "application/json")
        metric_depth.close()
        metric_preview = self.client.get("/api/artifacts/metric_depth_ring01_preview")
        self.assertEqual(metric_preview.status_code, 200)
        self.assertEqual(metric_preview.mimetype, "image/png")
        metric_preview.close()
        normal_report = self.client.get("/api/artifacts/surface_normal_report")
        self.assertEqual(normal_report.status_code, 200)
        self.assertEqual(normal_report.mimetype, "application/json")
        normal_report.close()
        normal_preview = self.client.get("/api/artifacts/surface_normal_ring01_preview")
        self.assertEqual(normal_preview.status_code, 200)
        self.assertEqual(normal_preview.mimetype, "image/png")
        normal_preview.close()
        readiness_report = self.client.get("/api/artifacts/cross_view_readiness_report")
        self.assertEqual(readiness_report.status_code, 200)
        self.assertEqual(readiness_report.mimetype, "application/json")
        readiness_report.close()
        vggt_preview = self.client.get("/api/artifacts/ring01_vggt_depth_preview")
        self.assertEqual(vggt_preview.status_code, 200)
        self.assertEqual(vggt_preview.mimetype, "image/png")
        vggt_preview.close()
        vggt_cloud = self.client.get("/api/artifacts/ring01_vggt_cloud_preview")
        self.assertEqual(vggt_cloud.status_code, 200)
        self.assertEqual(vggt_cloud.mimetype, "image/png")
        vggt_cloud.close()
        vggt_review = self.client.get("/api/artifacts/ring01_vggt_visual_review")
        self.assertEqual(vggt_review.status_code, 200)
        self.assertEqual(vggt_review.mimetype, "application/json")
        vggt_review.close()
        external_preview = self.client.get("/api/artifacts/external_syr_preview")
        self.assertEqual(external_preview.status_code, 200)
        self.assertEqual(external_preview.mimetype, "image/jpeg")
        external_preview.close()
        upload_doc = self.client.get("/api/artifacts/upload_auditor_doc")
        self.assertEqual(upload_doc.status_code, 200)
        self.assertEqual(upload_doc.mimetype, "text/markdown")
        upload_doc.close()
        script.close()
        page.close()

    def test_phase1_review_routes_are_scoped_and_fail_closed(self) -> None:
        manifest = self.client.get("/api/phase1/review")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(len(manifest.get_json()["views"]), 5)
        manifest.close()

        source = self.client.get("/api/phase1/assets/front/source")
        self.assertEqual(source.status_code, 200)
        self.assertEqual(source.mimetype, "image/png")
        source.close()

        proposal = self.client.get(
            "/api/phase1/assets/front/proposal?label=prongs"
        )
        self.assertEqual(proposal.status_code, 200)
        self.assertEqual(proposal.mimetype, "image/png")
        proposal.close()

        invalid = self.client.post(
            "/api/phase1/reviews/label",
            json={
                "view": "../../etc",
                "label": "prongs",
                "decision": "approved",
                "reviewer": "test",
            },
        )
        self.assertEqual(invalid.status_code, 400)

        stl1 = self.client.get("/api/phase1/stl1/review")
        self.assertEqual(stl1.status_code, 200)
        self.assertEqual(stl1.get_json()["gate"]["object_count"], 24)
        self.assertEqual(len(stl1.get_json()["objects"]), 24)
        stl1.close()

        object_review = self.client.get("/api/phase1/stl1/review/ring_001")
        self.assertEqual(object_review.status_code, 200)
        self.assertEqual(len(object_review.get_json()["object"]["views"]), 5)
        object_review.close()

        evidence = self.client.get(
            "/api/phase1/stl1/assets/ring_001/front/silhouette"
        )
        self.assertEqual(evidence.status_code, 200)
        self.assertEqual(evidence.mimetype, "image/png")
        evidence.close()

        invalid_stl1 = self.client.post(
            "/api/phase1/stl1/reviews/evidence",
            json={
                "object_id": "../../etc",
                "view": "front",
                "evidence_type": "silhouette",
                "decision": "approved",
                "reviewer": "test",
            },
        )
        self.assertEqual(invalid_stl1.status_code, 400)

    def test_object_and_asset_routes(self) -> None:
        detail = self.client.get("/api/objects/ring_001")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(len(detail.get_json()["views"]), 5)
        image = self.client.get("/api/objects/ring_001/assets/source?view=front")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/png")
        image.close()
        depth = self.client.get("/api/artifacts/phase1_depth_review")
        self.assertEqual(depth.status_code, 200)
        self.assertEqual(depth.mimetype, "image/png")
        depth.close()
        front_audit = self.client.get("/api/artifacts/ring01_front_depth_audit")
        self.assertEqual(front_audit.status_code, 200)
        self.assertEqual(front_audit.mimetype, "image/png")
        front_audit.close()
        pixel_audit = self.client.get("/api/artifacts/ring01_pixel_feature_audit")
        self.assertEqual(pixel_audit.status_code, 200)
        self.assertEqual(pixel_audit.mimetype, "image/png")
        pixel_audit.close()
        dataset_depth = self.client.get(
            "/api/objects/ring_001/assets/depth?view=front"
        )
        self.assertEqual(dataset_depth.status_code, 200)
        self.assertEqual(dataset_depth.mimetype, "image/png")
        dataset_depth.close()
        ensemble = self.client.get(
            "/api/objects/ring_001/assets/depth_ensemble?view=front"
        )
        self.assertEqual(ensemble.status_code, 200)
        self.assertEqual(ensemble.mimetype, "image/png")
        ensemble.close()
        transform_uncertainty = self.client.get(
            "/api/objects/ring_001/assets/depth_transform_uncertainty?view=front"
        )
        self.assertEqual(transform_uncertainty.status_code, 200)
        self.assertEqual(transform_uncertainty.mimetype, "image/png")
        transform_uncertainty.close()
        depth_overlay = self.client.get(
            "/api/objects/ring_001/assets/depth_overlay?view=front"
        )
        self.assertEqual(depth_overlay.status_code, 200)
        self.assertEqual(depth_overlay.mimetype, "image/png")
        depth_overlay.close()
        review = self.client.get(
            "/api/objects/ring_001/assets/phase1_depth_review"
        )
        self.assertEqual(review.status_code, 200)
        self.assertEqual(review.mimetype, "image/jpeg")
        review.close()
        complete_audit = self.client.get(
            "/api/objects/ring_001/assets/phase1_complete_audit"
        )
        self.assertEqual(complete_audit.status_code, 200)
        self.assertEqual(complete_audit.mimetype, "image/png")
        complete_audit.close()
        overview = self.client.get("/api/artifacts/dataset_depth_overview")
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.mimetype, "image/jpeg")
        overview.close()
        complete_overview = self.client.get(
            "/api/artifacts/dataset_complete_phase1_overview"
        )
        self.assertEqual(complete_overview.status_code, 200)
        self.assertEqual(complete_overview.mimetype, "image/png")
        complete_overview.close()
        multiring_report = self.client.get(
            "/api/artifacts/vggt_multiring_validation"
        )
        self.assertEqual(multiring_report.status_code, 200)
        multiring_report.close()
        multiring_preview = self.client.get(
            "/api/artifacts/vggt_ring_007_reprojection"
        )
        self.assertEqual(multiring_preview.status_code, 200)
        self.assertEqual(multiring_preview.mimetype, "image/png")
        multiring_preview.close()
        refinement_preview = self.client.get(
            "/api/artifacts/vggt_multiring_refinement_preview"
        )
        self.assertEqual(refinement_preview.status_code, 200)
        self.assertEqual(refinement_preview.mimetype, "image/jpeg")
        refinement_preview.close()
        semantic_report = self.client.get(
            "/api/artifacts/semantic_camera_priors_report"
        )
        self.assertEqual(semantic_report.status_code, 200)
        self.assertEqual(semantic_report.mimetype, "application/json")
        semantic_report.close()
        semantic_preview = self.client.get(
            "/api/artifacts/semantic_ring_023_holdout"
        )
        self.assertEqual(semantic_preview.status_code, 200)
        self.assertEqual(semantic_preview.mimetype, "image/png")
        semantic_preview.close()
        resolution_report = self.client.get(
            "/api/artifacts/vggt_resolution_ladder_report"
        )
        self.assertEqual(resolution_report.status_code, 200)
        self.assertEqual(resolution_report.mimetype, "application/json")
        resolution_report.close()
        resolution_chart = self.client.get(
            "/api/artifacts/vggt_resolution_ladder_chart"
        )
        self.assertEqual(resolution_chart.status_code, 200)
        self.assertEqual(resolution_chart.mimetype, "image/png")
        resolution_chart.close()
        transfer_preview = self.client.get(
            "/api/artifacts/vggt_view_transfer_ring_007"
        )
        self.assertEqual(transfer_preview.status_code, 200)
        self.assertEqual(transfer_preview.mimetype, "image/png")
        transfer_preview.close()
        completion_report = self.client.get(
            "/api/artifacts/vggt_visual_hull_completion_report"
        )
        self.assertEqual(completion_report.status_code, 200)
        self.assertEqual(completion_report.mimetype, "application/json")
        completion_report.close()
        completion_preview = self.client.get(
            "/api/artifacts/vggt_visual_hull_ring_020"
        )
        self.assertEqual(completion_preview.status_code, 200)
        self.assertEqual(completion_preview.mimetype, "image/png")
        completion_preview.close()
        surface_report = self.client.get(
            "/api/artifacts/vggt_surface_aware_hull_report"
        )
        self.assertEqual(surface_report.status_code, 200)
        self.assertEqual(surface_report.mimetype, "application/json")
        surface_report.close()
        surface_preview = self.client.get(
            "/api/artifacts/vggt_surface_hull_ring_020"
        )
        self.assertEqual(surface_preview.status_code, 200)
        self.assertEqual(surface_preview.mimetype, "image/png")
        surface_preview.close()
        surface_test_preview = self.client.get(
            "/api/artifacts/vggt_surface_hull_ring_015"
        )
        self.assertEqual(surface_test_preview.status_code, 200)
        self.assertEqual(surface_test_preview.mimetype, "image/png")
        surface_test_preview.close()
        signed_depth_report = self.client.get(
            "/api/artifacts/vggt_signed_depth_carving_report"
        )
        self.assertEqual(signed_depth_report.status_code, 200)
        self.assertEqual(signed_depth_report.mimetype, "application/json")
        signed_depth_report.close()
        signed_depth_test_preview = self.client.get(
            "/api/artifacts/vggt_signed_depth_ring_015"
        )
        self.assertEqual(signed_depth_test_preview.status_code, 200)
        self.assertEqual(signed_depth_test_preview.mimetype, "image/png")
        signed_depth_test_preview.close()
        normal_report = self.client.get(
            "/api/artifacts/vggt_normal_consistency_report"
        )
        self.assertEqual(normal_report.status_code, 200)
        self.assertEqual(normal_report.mimetype, "application/json")
        normal_report.close()
        normal_preview = self.client.get(
            "/api/artifacts/vggt_normal_consistency_ring_015"
        )
        self.assertEqual(normal_preview.status_code, 200)
        self.assertEqual(normal_preview.mimetype, "image/png")
        normal_preview.close()
        aperture_report = self.client.get(
            "/api/artifacts/vggt_negative_space_report"
        )
        self.assertEqual(aperture_report.status_code, 200)
        self.assertEqual(aperture_report.mimetype, "application/json")
        aperture_report.close()
        aperture_preview = self.client.get(
            "/api/artifacts/vggt_negative_space_ring_015"
        )
        self.assertEqual(aperture_preview.status_code, 200)
        self.assertEqual(aperture_preview.mimetype, "image/png")
        aperture_preview.close()
        camera_alignment_report = self.client.get(
            "/api/artifacts/vggt_source_camera_alignment_report"
        )
        self.assertEqual(camera_alignment_report.status_code, 200)
        self.assertEqual(camera_alignment_report.mimetype, "application/json")
        camera_alignment_report.close()
        camera_alignment_preview = self.client.get(
            "/api/artifacts/vggt_source_camera_alignment_ring_015"
        )
        self.assertEqual(camera_alignment_preview.status_code, 200)
        self.assertEqual(camera_alignment_preview.mimetype, "image/png")
        camera_alignment_preview.close()
        rigid_registration_report = self.client.get(
            "/api/artifacts/vggt_rigid_source_registration_report"
        )
        self.assertEqual(rigid_registration_report.status_code, 200)
        self.assertEqual(rigid_registration_report.mimetype, "application/json")
        rigid_registration_report.close()
        rigid_registration_preview = self.client.get(
            "/api/artifacts/vggt_rigid_registration_ring_020_medium"
        )
        self.assertEqual(rigid_registration_preview.status_code, 200)
        self.assertEqual(rigid_registration_preview.mimetype, "image/png")
        rigid_registration_preview.close()
        track_report = self.client.get(
            "/api/artifacts/vggt_track_evidence_report"
        )
        self.assertEqual(track_report.status_code, 200)
        self.assertEqual(track_report.mimetype, "application/json")
        track_report.close()
        track_registration_report = self.client.get(
            "/api/artifacts/vggt_track_guided_registration_report"
        )
        self.assertEqual(track_registration_report.status_code, 200)
        self.assertEqual(track_registration_report.mimetype, "application/json")
        track_registration_report.close()
        track_preview = self.client.get("/api/artifacts/vggt_tracks_ring_020")
        self.assertEqual(track_preview.status_code, 200)
        self.assertEqual(track_preview.mimetype, "image/png")
        track_preview.close()
        track_registration_preview = self.client.get(
            "/api/artifacts/vggt_track_registration_ring_007"
        )
        self.assertEqual(track_registration_preview.status_code, 200)
        self.assertEqual(track_registration_preview.mimetype, "image/png")
        track_registration_preview.close()
        dense_track_report = self.client.get(
            "/api/artifacts/vggt_dense_track_evidence_report"
        )
        self.assertEqual(dense_track_report.status_code, 200)
        self.assertEqual(dense_track_report.mimetype, "application/json")
        dense_track_report.close()
        dense_cycle_report = self.client.get(
            "/api/artifacts/vggt_dense_cycle_registration_report"
        )
        self.assertEqual(dense_cycle_report.status_code, 200)
        self.assertEqual(dense_cycle_report.mimetype, "application/json")
        dense_cycle_report.close()
        dense_cycle_preview = self.client.get(
            "/api/artifacts/vggt_dense_cycle_ring_015"
        )
        self.assertEqual(dense_cycle_preview.status_code, 200)
        self.assertEqual(dense_cycle_preview.mimetype, "image/png")
        dense_cycle_preview.close()
        projection_noise_report = self.client.get(
            "/api/artifacts/vggt_projection_noise_report"
        )
        self.assertEqual(projection_noise_report.status_code, 200)
        self.assertEqual(projection_noise_report.mimetype, "application/json")
        projection_noise_report.close()
        projection_noise_preview = self.client.get(
            "/api/artifacts/vggt_projection_noise_ring_015"
        )
        self.assertEqual(projection_noise_preview.status_code, 200)
        self.assertEqual(projection_noise_preview.mimetype, "image/png")
        projection_noise_preview.close()

    def test_invalid_routes_fail_closed(self) -> None:
        self.assertEqual(
            self.client.get("/api/phase4/graphs/not-a-graph").status_code,
            404,
        )
        self.assertEqual(self.client.get("/api/objects/ring_999").status_code, 404)
        self.assertEqual(
            self.client.get("/api/objects/ring_001/assets/source?view=../../etc").status_code,
            404,
        )

    def test_phase4_graph_and_dry_run_edit_routes(self) -> None:
        ring01 = self.client.get("/api/phase4/graphs/ring01")
        self.assertEqual(ring01.status_code, 200)
        self.assertEqual(len(ring01.get_json()["nodes"]), 10)
        ring01.close()
        prong_audit = self.client.get("/api/artifacts/phase4_prong_instance_audit")
        self.assertEqual(prong_audit.status_code, 200)
        self.assertEqual(prong_audit.mimetype, "image/png")
        prong_audit.close()
        track_audit = self.client.get("/api/artifacts/phase4_component_track_audit")
        self.assertEqual(track_audit.status_code, 200)
        self.assertEqual(track_audit.mimetype, "image/png")
        track_audit.close()
        candidate_audit = self.client.get(
            "/api/artifacts/phase4_stl1_candidate_audit"
        )
        self.assertEqual(candidate_audit.status_code, 200)
        self.assertEqual(candidate_audit.mimetype, "image/png")
        candidate_audit.close()
        candidate_report = self.client.get(
            "/api/artifacts/phase4_stl1_candidate_report"
        )
        self.assertEqual(candidate_report.status_code, 200)
        self.assertEqual(candidate_report.mimetype, "application/json")
        candidate_report.close()
        exact_cad_audit = self.client.get(
            "/api/artifacts/phase4_exact_cad_locality_audit"
        )
        self.assertEqual(exact_cad_audit.status_code, 200)
        self.assertEqual(exact_cad_audit.mimetype, "image/png")
        exact_cad_audit.close()
        exact_cad_report = self.client.get(
            "/api/artifacts/phase4_exact_cad_locality_report"
        )
        self.assertEqual(exact_cad_report.status_code, 200)
        self.assertEqual(exact_cad_report.mimetype, "application/json")
        exact_cad_report.close()
        phase5_audit = self.client.get(
            "/api/artifacts/phase5_exact_cad_v2_audit"
        )
        self.assertEqual(phase5_audit.status_code, 200)
        self.assertEqual(phase5_audit.mimetype, "image/png")
        phase5_audit.close()
        phase5_dataset_audit = self.client.get(
            "/api/artifacts/phase5_stl1_ring_parameter_audit"
        )
        self.assertEqual(phase5_dataset_audit.status_code, 200)
        self.assertEqual(phase5_dataset_audit.mimetype, "image/png")
        phase5_dataset_audit.close()
        phase5_dataset_report = self.client.get(
            "/api/artifacts/phase5_stl1_ring_parameter_report"
        )
        self.assertEqual(phase5_dataset_report.status_code, 200)
        self.assertEqual(phase5_dataset_report.mimetype, "application/json")
        phase5_dataset_report.close()
        phase5_ring_body_audit = self.client.get(
            "/api/artifacts/phase5_stl1_ring_body_fit_audit"
        )
        self.assertEqual(phase5_ring_body_audit.status_code, 200)
        self.assertEqual(phase5_ring_body_audit.mimetype, "image/png")
        phase5_ring_body_audit.close()
        phase5_ring_body_step = self.client.get(
            "/api/artifacts/phase5_ring004_ring_body_step"
        )
        self.assertEqual(phase5_ring_body_step.status_code, 200)
        self.assertEqual(phase5_ring_body_step.mimetype, "application/octet-stream")
        phase5_ring_body_step.close()
        phase5_head_audit = self.client.get(
            "/api/artifacts/phase5_stl1_head_support_fit_audit"
        )
        self.assertEqual(phase5_head_audit.status_code, 200)
        self.assertEqual(phase5_head_audit.mimetype, "image/png")
        phase5_head_audit.close()
        phase5_head_preview = self.client.get(
            "/api/artifacts/phase5_ring004_head_support_preview"
        )
        self.assertEqual(phase5_head_preview.status_code, 200)
        self.assertEqual(phase5_head_preview.mimetype, "image/png")
        phase5_head_preview.close()
        phase5_head_step = self.client.get(
            "/api/artifacts/phase5_ring004_head_support_step"
        )
        self.assertEqual(phase5_head_step.status_code, 200)
        self.assertEqual(phase5_head_step.mimetype, "application/octet-stream")
        phase5_head_step.close()
        phase5_openwork_audit = self.client.get(
            "/api/artifacts/phase5_stl1_openwork_fit_audit"
        )
        self.assertEqual(phase5_openwork_audit.status_code, 200)
        self.assertEqual(phase5_openwork_audit.mimetype, "image/png")
        phase5_openwork_audit.close()
        phase5_openwork_preview = self.client.get(
            "/api/artifacts/phase5_ring004_openwork_preview"
        )
        self.assertEqual(phase5_openwork_preview.status_code, 200)
        self.assertEqual(phase5_openwork_preview.mimetype, "image/png")
        phase5_openwork_preview.close()
        phase5_openwork_step = self.client.get(
            "/api/artifacts/phase5_ring004_openwork_step"
        )
        self.assertEqual(phase5_openwork_step.status_code, 200)
        self.assertEqual(
            phase5_openwork_step.mimetype, "application/octet-stream"
        )
        phase5_openwork_step.close()
        phase5_relief_audit = self.client.get(
            "/api/artifacts/phase5_stl1_relief_evidence_audit"
        )
        self.assertEqual(phase5_relief_audit.status_code, 200)
        self.assertEqual(phase5_relief_audit.mimetype, "image/png")
        phase5_relief_audit.close()
        phase5_relief_review = self.client.get(
            "/api/artifacts/phase5_ring004_relief_review"
        )
        self.assertEqual(phase5_relief_review.status_code, 200)
        self.assertEqual(phase5_relief_review.mimetype, "image/png")
        phase5_relief_review.close()
        phase5_relief_obj = self.client.get(
            "/api/artifacts/phase5_ring004_relief_obj"
        )
        self.assertEqual(phase5_relief_obj.status_code, 200)
        self.assertEqual(
            phase5_relief_obj.mimetype, "application/octet-stream"
        )
        phase5_relief_obj.close()
        phase5_relief_track_audit = self.client.get(
            "/api/artifacts/phase5_relief_track_audit"
        )
        self.assertEqual(phase5_relief_track_audit.status_code, 200)
        self.assertEqual(phase5_relief_track_audit.mimetype, "image/png")
        phase5_relief_track_audit.close()
        phase5_relief_track_review = self.client.get(
            "/api/artifacts/phase5_ring004_relief_track_review"
        )
        self.assertEqual(phase5_relief_track_review.status_code, 200)
        self.assertEqual(phase5_relief_track_review.mimetype, "image/png")
        phase5_relief_track_review.close()
        phase5_full_relief_track_audit = self.client.get(
            "/api/artifacts/phase5_full_relief_track_audit"
        )
        self.assertEqual(phase5_full_relief_track_audit.status_code, 200)
        self.assertEqual(phase5_full_relief_track_audit.mimetype, "image/png")
        phase5_full_relief_track_audit.close()
        phase5_full_relief_track_report = self.client.get(
            "/api/artifacts/phase5_full_relief_track_report"
        )
        self.assertEqual(phase5_full_relief_track_report.status_code, 200)
        self.assertEqual(phase5_full_relief_track_report.mimetype, "application/json")
        phase5_full_relief_track_report.close()
        phase5_full_ring004_relief_track_review = self.client.get(
            "/api/artifacts/phase5_full_ring004_relief_track_review"
        )
        self.assertEqual(phase5_full_ring004_relief_track_review.status_code, 200)
        self.assertEqual(
            phase5_full_ring004_relief_track_review.mimetype, "image/png"
        )
        phase5_full_ring004_relief_track_review.close()
        phase5_dense_train_audit = self.client.get(
            "/api/artifacts/phase5_dense_relief_train_audit"
        )
        self.assertEqual(phase5_dense_train_audit.status_code, 200)
        self.assertEqual(phase5_dense_train_audit.mimetype, "image/png")
        phase5_dense_train_audit.close()
        phase5_dense_validation_audit = self.client.get(
            "/api/artifacts/phase5_dense_relief_validation_audit"
        )
        self.assertEqual(phase5_dense_validation_audit.status_code, 200)
        self.assertEqual(phase5_dense_validation_audit.mimetype, "image/png")
        phase5_dense_validation_audit.close()
        phase5_dense_validation_report = self.client.get(
            "/api/artifacts/phase5_dense_relief_validation_report"
        )
        self.assertEqual(phase5_dense_validation_report.status_code, 200)
        self.assertEqual(
            phase5_dense_validation_report.mimetype, "application/json"
        )
        phase5_dense_validation_report.close()
        phase5_normal_audit = self.client.get(
            "/api/artifacts/phase5_normal_corroboration_audit"
        )
        self.assertEqual(phase5_normal_audit.status_code, 200)
        self.assertEqual(phase5_normal_audit.mimetype, "image/png")
        phase5_normal_audit.close()
        phase5_normal_report = self.client.get(
            "/api/artifacts/phase5_normal_corroboration_report"
        )
        self.assertEqual(phase5_normal_report.status_code, 200)
        self.assertEqual(phase5_normal_report.mimetype, "application/json")
        phase5_normal_report.close()
        phase5_visibility_audit = self.client.get(
            "/api/artifacts/phase5_visibility_graph_audit"
        )
        self.assertEqual(phase5_visibility_audit.status_code, 200)
        self.assertEqual(phase5_visibility_audit.mimetype, "image/png")
        phase5_visibility_audit.close()
        phase5_visibility_report = self.client.get(
            "/api/artifacts/phase5_visibility_graph_report"
        )
        self.assertEqual(phase5_visibility_report.status_code, 200)
        self.assertEqual(phase5_visibility_report.mimetype, "application/json")
        phase5_visibility_report.close()
        phase5_ring004_visibility = self.client.get(
            "/api/artifacts/phase5_ring004_visibility_graph"
        )
        self.assertEqual(phase5_ring004_visibility.status_code, 200)
        self.assertEqual(
            phase5_ring004_visibility.mimetype, "application/json"
        )
        phase5_ring004_visibility.close()
        for artifact in (
            "phase5_ring004_validation_normals",
            "phase5_ring012_validation_normals",
            "phase5_ring020_validation_normals",
        ):
            response = self.client.get(f"/api/artifacts/{artifact}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
            response.close()
        candidate_summary = self.client.get("/api/phase4/stl1-candidates")
        self.assertEqual(candidate_summary.status_code, 200)
        self.assertEqual(
            candidate_summary.get_json()["gate"]["candidate_count"], 266
        )
        candidate_summary.close()
        candidate_object = self.client.get(
            "/api/phase4/stl1-candidates/ring_014"
        )
        self.assertEqual(candidate_object.status_code, 200)
        object_payload = candidate_object.get_json()
        self.assertEqual(
            len(object_payload["object"]["views"]["front"]["candidates"]), 5
        )
        candidate_id = object_payload["object"]["views"]["front"][
            "candidates"
        ][0]["candidate_id"]
        candidate_object.close()
        for kind in ("source", "overlay"):
            asset = self.client.get(
                f"/api/phase4/stl1-candidates/ring_014/front/{kind}"
            )
            self.assertEqual(asset.status_code, 200)
            self.assertEqual(asset.mimetype, "image/png")
            asset.close()
        mask = self.client.get(
            "/api/phase4/stl1-candidates/ring_014/front/mask"
            f"?candidate_id={candidate_id}"
        )
        self.assertEqual(mask.status_code, 200)
        self.assertEqual(mask.mimetype, "image/png")
        mask.close()
        unidentified = self.client.post(
            "/api/phase4/stl1-candidates/review",
            json={
                "object_id": "ring_014",
                "view": "front",
                "candidate_id": candidate_id,
                "semantic_label": "aperture",
                "decision": "approved",
            },
        )
        self.assertEqual(unidentified.status_code, 400)
        unidentified.close()
        synthetic = self.client.get("/api/phase4/graphs/synthetic-ten-stone")
        self.assertEqual(synthetic.status_code, 200)
        self.assertEqual(
            sum(
                node["semantic_class"] == "round_gemstone"
                for node in synthetic.get_json()["nodes"]
            ),
            10,
        )
        synthetic.close()
        preview = self.client.post(
            "/api/phase4/graphs/synthetic-ten-stone/edit-preview",
            json={
                "component_id": "stone_007",
                "operation": "resize",
                "parameters": {"diameter": 2.8, "depth": 1.5},
            },
        )
        self.assertEqual(preview.status_code, 200)
        payload = preview.get_json()
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["graph_state_only"])
        self.assertFalse(payload["geometry_generated"])
        self.assertTrue(payload["audit"]["affected_set_exact"])
        self.assertTrue(payload["audit"]["unrelated_geometry_state_unchanged"])
        preview.close()
        rejected = self.client.post(
            "/api/phase4/graphs/synthetic-ten-stone/edit-preview",
            json={"component_id": "stone_999", "operation": "resize"},
        )
        self.assertEqual(rejected.status_code, 400)


if __name__ == "__main__":
    unittest.main()
