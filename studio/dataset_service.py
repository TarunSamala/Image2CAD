"""Service layer for retained dataset evidence and scoped Phase 1 reviews.

Artifact browsing is read-only. The only mutations accepted here are explicit,
identified Phase 1 human-review decisions written to the versioned manifest.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from pipeline.component_constraint_graph import (
    ComponentGraphError,
    apply_local_edit,
)
from pipeline.phase4_candidate_review import (
    apply_candidate_review as apply_phase4_candidate_review,
    load_manifest as load_phase4_candidate_review,
)
from pipeline.phase1_review import (
    LABELS as PHASE1_LABELS,
    VIEWS as PHASE1_VIEWS,
    apply_camera_review,
    apply_identity_review,
    apply_label_review,
)
from pipeline.stl1_phase1_review import (
    EVIDENCE_TYPES as STL1_PHASE1_EVIDENCE_TYPES,
    VIEWS as STL1_PHASE1_VIEWS,
    apply_evidence_review as apply_stl1_evidence_review,
    apply_object_review as apply_stl1_object_review,
    apply_view_review as apply_stl1_view_review,
    load_manifest as load_stl1_phase1_manifest,
    save_corrected_silhouette,
)


VIEW_ORDER = ("front", "top", "iso", "lsv", "rsv")
VIEW_LABELS = {
    "front": "Front",
    "top": "Top",
    "iso": "Isometric",
    "lsv": "Left side",
    "rsv": "Right side",
}


class DatasetError(RuntimeError):
    """Base exception for unavailable or invalid dataset state."""


class ObjectNotFoundError(DatasetError):
    """Raised when an object ID is not present in the manifest."""


class AssetNotFoundError(DatasetError):
    """Raised when a requested artifact is unavailable."""


class DatasetRepository:
    """Index the prepared dataset and retained phase artifacts.

    All paths returned by :meth:`asset_path` are checked to remain under the
    repository root.  The service never executes a phase or mutates evidence.
    """

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
        self.prepared_root = self.repo_root / "dataset" / "prepared_v1"
        self.phase_root = self.repo_root / "dataset" / "phase_runs" / "v1"

        self.dataset_report = self._read_json(self.prepared_root / "dataset_report.json")
        self.phase_summary = self._read_json(self.phase_root / "summary.json")
        self.audit_report = self._read_json(
            self.phase_root / "phase_audits" / "phase_audit_report.json"
        )
        self.phase3_report = self._read_json(
            self.phase_root / "phase3" / "visual_hull" / "phase3_batch_report.json"
        )
        self.phase3_geometry = self._read_json(
            self.phase_root / "phase3_geometry_benchmark_v1" / "report.json"
        )
        self.phase4_component_graph = self._read_json(
            self.repo_root / "data" / "phase4_component_graph_v3" / "report.json"
        )
        self.phase4_prong_instances = self._read_json(
            self.repo_root / "data" / "phase4_prong_instances_v1" / "report.json"
        )
        self.phase4_component_tracks = self._read_json(
            self.repo_root / "data" / "phase4_component_tracks_v1" / "report.json"
        )
        self.phase4_stl1_candidates = self._read_json(
            self.repo_root
            / "data"
            / "phase4_stl1_component_candidates_v1"
            / "report.json"
        )
        _, self.phase4_candidate_review = load_phase4_candidate_review(
            self.repo_root
        )
        self.phase4_ring01_graph = self._read_json(
            self.repo_root
            / "data"
            / "phase4_component_graph_v3"
            / "ring01_component_graph.json"
        )
        self.phase4_synthetic_graph = self._read_json(
            self.repo_root
            / "data"
            / "phase4_component_graph_v3"
            / "synthetic_ten_stone_graph.json"
        )
        self.phase1_features = self._read_json(self.phase_root / "phase1_features.json")
        self.phase1_depth = self._read_json(
            self.phase_root / "phase1_depth_anything_v2" / "report.json"
        )
        self.phase1_complete = self._read_json(
            self.phase_root / "phase1_complete_features_v1" / "report.json"
        )
        self.phase0_environment = self._read_json(
            self.repo_root
            / "docs"
            / "bible_implementation"
            / "phase_0_reproducibility"
            / "environment_manifest.json"
        )
        self.phase0_licenses = self._read_json(
            self.repo_root
            / "docs"
            / "bible_implementation"
            / "phase_0_reproducibility"
            / "dependency_license_manifest.json"
        )
        self.phase1_ground_truth = self._read_json(
            self.repo_root / "data" / "ring01_ground_truth_v1" / "manifest.json"
        )
        self.ring01_review_notes = self._read_json(
            self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "review"
            / "ring01_review_notes.json"
        )
        self.phase2_paired = self._read_json(
            self.repo_root / "dataset" / "paired_cad_v1" / "report.json"
        )
        self.phase2_image_only = self._read_json(
            self.repo_root
            / "data"
            / "ring01_image_only"
            / "reflection_correction_v1"
            / "report.json"
        )
        self.phase2_machine_review = self._read_json(
            self.repo_root
            / "data"
            / "ring01_image_only"
            / "machine_review_v1"
            / "report.json"
        )
        self.metric_depth_consensus = self._read_json(
            self.phase_root
            / "image_only_phase2_metric_depth_consensus_v1"
            / "report.json"
        )
        self.surface_normal_evidence = self._read_json(
            self.phase_root / "surface_normal_evidence_v1" / "report.json"
        )
        self.cross_view_readiness = self._read_json(
            self.phase_root / "cross_view_backend_readiness_v1" / "report.json"
        )
        self.ring01_vggt_validation = self._read_json(
            self.phase_root / "ring01_vggt_dense_224_v1" / "validation_report.json"
        )
        self.ring01_vggt_visual_review = self._read_json(
            self.phase_root / "ring01_vggt_dense_224_v1" / "machine_visual_review.json"
        )
        self.vggt_multiring_validation = self._read_json(
            self.phase_root / "vggt_multiring_224_v1" / "validation_report.json"
        )
        self.vggt_multiring_refinement = self._read_json(
            self.phase_root
            / "vggt_multiring_refinement_comparison_v1"
            / "report.json"
        )
        self.semantic_camera_priors = self._read_json(
            self.phase_root / "semantic_camera_priors_v1" / "report.json"
        )
        self.semantic_camera_refinement = self._read_json(
            self.phase_root
            / "vggt_semantic_camera_refinement_v1"
            / "report.json"
        )
        self.vggt_view_transfer = self._read_json(
            self.phase_root / "vggt_view_transfer_v1" / "report.json"
        )
        self.vggt_resolution_ladder = self._read_json(
            self.phase_root / "vggt_resolution_ladder_v1" / "report.json"
        )
        self.vggt_visual_hull_completion = self._read_json(
            self.phase_root / "vggt_visual_hull_completion_v1" / "report.json"
        )
        self.vggt_surface_aware_hull = self._read_json(
            self.phase_root / "vggt_surface_aware_hull_v1" / "report.json"
        )
        self.vggt_signed_depth_carving = self._read_json(
            self.phase_root / "vggt_signed_depth_carving_v1" / "report.json"
        )
        self.vggt_normal_consistency = self._read_json(
            self.phase_root / "vggt_normal_consistency_v1" / "report.json"
        )
        self.vggt_negative_space = self._read_json(
            self.phase_root / "vggt_negative_space_v1" / "report.json"
        )
        self.vggt_source_camera_alignment = self._read_json(
            self.phase_root
            / "vggt_source_camera_alignment_v1"
            / "report.json"
        )
        self.vggt_rigid_source_registration = self._read_json(
            self.phase_root
            / "vggt_rigid_source_registration_v1"
            / "report.json"
        )
        self.vggt_track_evidence = self._read_json(
            self.phase_root / "vggt_track_evidence_v1" / "report.json"
        )
        self.vggt_track_guided_registration = self._read_json(
            self.phase_root
            / "vggt_track_guided_registration_v1"
            / "report.json"
        )
        self.vggt_dense_track_evidence = self._read_json(
            self.phase_root / "vggt_dense_track_evidence_v1" / "report.json"
        )
        self.vggt_dense_cycle_registration = self._read_json(
            self.phase_root
            / "vggt_dense_cycle_consistent_registration_v1"
            / "report.json"
        )
        self.vggt_projection_noise = self._read_json(
            self.phase_root / "vggt_projection_noise_v1" / "report.json"
        )
        self.external_ring_report = self._read_json(
            self.repo_root / "dataset" / "external_ring_v1" / "report.json"
        )
        self.external_ring_registry = self._read_json(
            self.repo_root / "dataset" / "external_sources" / "registry.json"
        )
        self.stl1_phase1_review_path, self.stl1_phase1_review = (
            load_stl1_phase1_manifest(self.repo_root)
        )
        self.phase1_depth_validation = self._read_json(
            self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "depth_validation_v1"
            / "report.json"
        )
        self.phase1_pixel_features = self._read_json(
            self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "pixel_features_v1"
            / "report.json"
        )
        self.records = self._read_manifest(self.prepared_root / "manifest.jsonl")
        self._records_by_id = {record["object_id"]: record for record in self.records}
        self._phase1_by_object = self._index_phase1_features()
        self._phase1_depth_by_object = {
            item["object_id"]: item for item in self.phase1_depth.get("objects", [])
        }
        self._phase1_complete_by_object = {
            item["object_id"]: item for item in self.phase1_complete.get("objects", [])
        }
        self._phase2_metrics = self._index_phase2_metrics()
        self._phase3_objects = {
            item["object_id"]: item for item in self.phase3_report.get("objects", [])
        }
        self._phase3_geometry_objects = {
            item["object_id"]: item
            for item in self.phase3_geometry.get("colmap", {}).get("objects", [])
        }
        self._validate_index()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise DatasetError(f"Required dataset report is missing: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Cannot read dataset report: {path}") from exc

    @staticmethod
    def _read_manifest(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise DatasetError(f"Dataset manifest is missing: {path}")
        records: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "object_id" not in record or "views" not in record:
                    raise DatasetError(f"Invalid manifest record on line {line_number}")
                records.append(record)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON in dataset manifest: {path}") from exc
        return sorted(records, key=lambda record: _natural_key(record["object_id"]))

    def _validate_index(self) -> None:
        expected = int(self.dataset_report.get("object_count", 0))
        if expected != len(self.records):
            raise DatasetError(
                f"Manifest/report object-count mismatch: {len(self.records)} != {expected}"
            )
        for record in self.records:
            missing = set(VIEW_ORDER) - set(record["views"])
            if missing:
                raise DatasetError(f"{record['object_id']} is missing views: {sorted(missing)}")
        missing_depth = set(self._records_by_id) - set(self._phase1_depth_by_object)
        if missing_depth:
            raise DatasetError(
                f"Phase 1 depth report is missing objects: {sorted(missing_depth)}"
            )
        missing_complete = set(self._records_by_id) - set(
            self._phase1_complete_by_object
        )
        if missing_complete:
            raise DatasetError(
                "Complete Phase 1 report is missing objects: "
                f"{sorted(missing_complete)}"
            )

    def _index_phase2_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
        indexed: dict[str, dict[str, dict[str, float]]] = {}
        for row in self.audit_report.get("views", []):
            indexed.setdefault(row["object_id"], {})[row["view"]] = {
                "iou": float(row["iou"]),
                "dice": float(row["dice"]),
                "boundary_f1_2px": float(row["boundary_f1_2px"]),
            }
        return indexed

    def _index_phase1_features(self) -> dict[str, dict[str, dict[str, Any]]]:
        indexed: dict[str, dict[str, dict[str, Any]]] = {}
        for row in self.phase1_features.get("views", []):
            indexed.setdefault(row["object_id"], {})[row["view"]] = {
                "foreground_fraction": float(row["foreground_fraction"]),
                "edge_fraction": float(row["edge_fraction"]),
                "contour_count": int(row["contour_count"]),
                "hole_count": int(row["hole_count"]),
                "horizontal_symmetry_iou": float(row["horizontal_symmetry_iou"]),
            }
        return indexed

    def _record(self, object_id: str) -> dict[str, Any]:
        try:
            return self._records_by_id[object_id]
        except KeyError as exc:
            raise ObjectNotFoundError(f"Unknown dataset object: {object_id}") from exc

    def _safe_path(self, relative_or_absolute: str | Path) -> Path:
        candidate = Path(relative_or_absolute)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise AssetNotFoundError("Requested path is outside the project repository") from exc
        if not resolved.is_file():
            raise AssetNotFoundError(f"Artifact is not available: {resolved.name}")
        return resolved

    def summary(self) -> dict[str, Any]:
        split_counts = deepcopy(self.dataset_report.get("split_counts", {}))
        phase3_objects = list(self._phase3_objects.values())
        return {
            "dataset": {
                "name": "STL-1",
                "object_count": len(self.records),
                "image_count": int(self.dataset_report.get("image_count", 0)),
                "views": list(VIEW_ORDER),
                "split_counts": split_counts,
                "prepared_size": int(self.dataset_report.get("image_size", 0)),
                "valid": bool(self.dataset_report.get("dataset_valid", False)),
            },
            "governance": {
                "project": "Lalitha / Image2CAD",
                "bible_release": "BASELINE ARCHITECTURE V1.0",
                "architecture": "evidence-driven jewellery reverse-engineering and CAD",
                "primary_scope": "rings",
                "capture_tier": "A",
                "evidence_class": "PROVEN-IN-PROJECT",
                "claim_level": "coarse non-metric research",
                "authoritative_output": "STEP / exact B-rep",
                "dataset_has_authoritative_step": False,
                "decision": "research_only",
            },
            "phase0": {
                "status": "complete",
                "source": deepcopy(self.phase0_environment.get("source", {})),
                "container": deepcopy(self.phase0_environment.get("container", {})),
                "runtime": deepcopy(self.phase0_environment.get("runtime", {})),
                "regression": deepcopy(self.phase0_environment.get("regression", {})),
                "known_limits": deepcopy(
                    self.phase0_environment.get("known_limits", [])
                ),
                "dependency_package_count": int(
                    self.phase0_licenses.get("package_count", 0)
                ),
                "dependency_unknown_license_count": int(
                    self.phase0_licenses.get("unknown_license_count", 0)
                ),
                "legal_status": self.phase0_licenses.get("legal_status"),
            },
            "phase1": {
                "stage": self.phase1_features.get("stage"),
                "view_count": int(self.phase1_features.get("view_count", 0)),
                "measurements_are_metric": bool(
                    self.phase1_features.get("measurements_are_metric", False)
                ),
                "checks": deepcopy(self.phase1_features.get("checks", {})),
                "depth_anything_v2": {
                    "status": self.phase1_depth.get("status"),
                    "decision": self.phase1_depth.get("decision"),
                    "model": deepcopy(self.phase1_depth.get("model", {})),
                    "coverage": deepcopy(self.phase1_depth.get("coverage", {})),
                    "checks": deepcopy(self.phase1_depth.get("checks", {})),
                    "runtime": deepcopy(self.phase1_depth.get("runtime", {})),
                    "metrics": deepcopy(self.phase1_depth.get("metrics", {})),
                    "phase1_gate": deepcopy(
                        self.phase1_depth.get("phase1_gate", {})
                    ),
                },
                "complete_features": {
                    "status": self.phase1_complete.get("status"),
                    "decision": self.phase1_complete.get("decision"),
                    "coverage": deepcopy(
                        self.phase1_complete.get("coverage", {})
                    ),
                    "runtime": deepcopy(self.phase1_complete.get("runtime", {})),
                    "checks": deepcopy(self.phase1_complete.get("checks", {})),
                    "feature_contract": deepcopy(
                        self.phase1_complete.get("feature_contract", {})
                    ),
                    "reconstruction_contract": deepcopy(
                        self.phase1_complete.get("reconstruction_contract", {})
                    ),
                    "depth_validation": deepcopy(
                        self.phase1_complete.get("depth_validation", {})
                    ),
                    "metric_depth": deepcopy(
                        self.phase1_complete.get("metric_depth", {})
                    ),
                },
            },
            "phase1_ground_truth": self._phase1_ground_truth_summary(),
            "stl1_phase1_review": self._stl1_phase1_review_summary(),
            "external_ring_datasets": {
                "status": self.external_ring_report.get("status"),
                "sample_count": int(self.external_ring_report.get("sample_count", 0)),
                "dataset_counts": deepcopy(
                    self.external_ring_report.get("dataset_counts", {})
                ),
                "ground_truth_mask_count": int(
                    self.external_ring_report.get("ground_truth_mask_count", 0)
                ),
                "approved_mask_count": int(
                    self.external_ring_report.get("approved_mask_count", 0)
                ),
                "review_required_mask_count": int(
                    self.external_ring_report.get("review_required_mask_count", 0)
                ),
                "mask_quality_counts": deepcopy(
                    self.external_ring_report.get("mask_quality_counts", {})
                ),
                "mask_exit_gate": deepcopy(
                    self.external_ring_report.get("mask_exit_gate", {})
                ),
                "metrics": deepcopy(self.external_ring_report.get("metrics", {})),
                "checks": deepcopy(self.external_ring_report.get("checks", {})),
                "source_count": len(self.external_ring_registry.get("sources", [])),
                "decision": "research_only_review_required",
            },
            "phase2": {
                **deepcopy(self.phase2_image_only),
                "ring01_carryover": deepcopy(self.ring01_review_notes),
                "machine_review": deepcopy(self.phase2_machine_review),
                "metric_depth_consensus": {
                    "stage": self.metric_depth_consensus.get("stage"),
                    "available_backends": deepcopy(
                        self.metric_depth_consensus.get("available_backends", [])
                    ),
                    "blocked_backends": deepcopy(
                        self.metric_depth_consensus.get("blocked_backends", {})
                    ),
                    "incomplete_backends": deepcopy(
                        self.metric_depth_consensus.get("incomplete_backends", {})
                    ),
                    "backend_coverage": deepcopy(
                        self.metric_depth_consensus.get("backend_coverage", {})
                    ),
                    "sample_count": self.metric_depth_consensus.get("sample_count", 0),
                    "summary": deepcopy(
                        self.metric_depth_consensus.get("summary", {})
                    ),
                    "same_view_alignment": deepcopy(
                        self.metric_depth_consensus.get("same_view_alignment", {})
                    ),
                    "geometry_alignment": deepcopy(
                        self.metric_depth_consensus.get("geometry_alignment", {})
                    ),
                    "claims": deepcopy(
                        self.metric_depth_consensus.get("claims", {})
                    ),
                    "visualization": deepcopy(
                        self.metric_depth_consensus.get("visualization", {})
                    ),
                    "decision": self.metric_depth_consensus.get("decision"),
                },
                "surface_normal_evidence": {
                    "stage": self.surface_normal_evidence.get("stage"),
                    "sample_count": self.surface_normal_evidence.get("sample_count", 0),
                    "object_count": self.surface_normal_evidence.get("object_count", 0),
                    "backends": deepcopy(
                        self.surface_normal_evidence.get("backends", {})
                    ),
                    "coordinate_contract": deepcopy(
                        self.surface_normal_evidence.get("coordinate_contract", {})
                    ),
                    "summary": deepcopy(
                        self.surface_normal_evidence.get("summary", {})
                    ),
                    "claims": deepcopy(
                        self.surface_normal_evidence.get("claims", {})
                    ),
                    "visualization": deepcopy(
                        self.surface_normal_evidence.get("visualization", {})
                    ),
                    "limitations": deepcopy(
                        self.surface_normal_evidence.get("limitations", [])
                    ),
                    "decision": self.surface_normal_evidence.get("decision"),
                },
            },
            "phase2_paired": {
                **deepcopy(self.phase2_paired),
                "applicability": "not_applicable_under_image_only_constraint",
            },
            "legacy_phase2": {
                "status": "pseudo-label self-consistency benchmark; not human-ground-truth accuracy",
                "test": deepcopy(self.phase_summary.get("test", {})),
                "metrics_by_split": deepcopy(self.audit_report.get("metrics_by_split", {})),
                "selected_threshold": self.phase_summary.get("selected_threshold"),
            },
            "phase3": {
                "status": self.phase3_report.get("phase3_status", "unavailable"),
                "object_count": len(phase3_objects),
                "all_meshes_watertight": bool(
                    self.phase3_report.get("all_meshes_watertight", False)
                ),
                "mean_reprojection_iou_by_split": deepcopy(
                    self.phase3_report.get("mean_reprojection_iou_by_split", {})
                ),
                "manufacturing_accuracy_validated": bool(
                    self.phase3_report.get("manufacturing_accuracy_validated", False)
                ),
                "artifact_authority": "derived research mesh; no authoritative STEP",
                "decision": "research_only",
                "geometry_benchmark": {
                    "stage": self.phase3_geometry.get("stage"),
                    "selected_object_count": self.phase3_geometry.get(
                        "selected_object_count", 0
                    ),
                    "colmap": {
                        "completed_sparse_model_count": self.phase3_geometry
                        .get("colmap", {})
                        .get("completed_sparse_model_count", 0),
                        "registered_all_five_views_count": self.phase3_geometry
                        .get("colmap", {})
                        .get("registered_all_five_views_count", 0),
                        "database_totals": deepcopy(
                            self.phase3_geometry
                            .get("colmap", {})
                            .get("database_totals", {})
                        ),
                    },
                    "backends": deepcopy(self.phase3_geometry.get("backends", {})),
                    "cross_view_readiness": deepcopy(self.cross_view_readiness),
                    "ring01_vggt_low_vram": {
                        **deepcopy(self.ring01_vggt_validation),
                        "machine_visual_review": deepcopy(
                            self.ring01_vggt_visual_review
                        ),
                    },
                    "vggt_multiring_low_vram": deepcopy(
                        self.vggt_multiring_validation
                    ),
                    "vggt_multiring_refinement": deepcopy(
                        self.vggt_multiring_refinement
                    ),
                    "semantic_camera_priors": deepcopy(
                        self.semantic_camera_priors
                    ),
                    "semantic_camera_refinement": deepcopy(
                        self.semantic_camera_refinement
                    ),
                    "vggt_view_transfer": deepcopy(self.vggt_view_transfer),
                    "vggt_resolution_ladder": deepcopy(
                        self.vggt_resolution_ladder
                    ),
                    "vggt_visual_hull_completion": deepcopy(
                        self.vggt_visual_hull_completion
                    ),
                    "vggt_surface_aware_hull": deepcopy(
                        self.vggt_surface_aware_hull
                    ),
                    "vggt_signed_depth_carving": deepcopy(
                        self.vggt_signed_depth_carving
                    ),
                    "vggt_normal_consistency": deepcopy(
                        self.vggt_normal_consistency
                    ),
                    "vggt_negative_space": deepcopy(
                        self.vggt_negative_space
                    ),
                    "vggt_source_camera_alignment": deepcopy(
                        self.vggt_source_camera_alignment
                    ),
                    "vggt_rigid_source_registration": deepcopy(
                        self.vggt_rigid_source_registration
                    ),
                    "vggt_track_evidence": deepcopy(
                        self.vggt_track_evidence
                    ),
                    "vggt_track_guided_registration": deepcopy(
                        self.vggt_track_guided_registration
                    ),
                    "vggt_dense_track_evidence": deepcopy(
                        self.vggt_dense_track_evidence
                    ),
                    "vggt_dense_cycle_registration": deepcopy(
                        self.vggt_dense_cycle_registration
                    ),
                    "vggt_projection_noise": deepcopy(
                        self.vggt_projection_noise
                    ),
                    "bible_gate": deepcopy(self.phase3_geometry.get("bible_gate", {})),
                    "claims": deepcopy(self.phase3_geometry.get("claims", {})),
                },
            },
            "phase4": {
                **deepcopy(self.phase4_component_graph),
                "prong_instances": deepcopy(self.phase4_prong_instances),
                "component_tracks": deepcopy(self.phase4_component_tracks),
                "stl1_component_candidates": deepcopy(
                    self.phase4_stl1_candidates
                ),
                "stl1_candidate_review": self._phase4_candidate_review_summary(),
            },
            "limitations": deepcopy(self.dataset_report.get("limitations", [])),
        }

    def phase4_graph(self, graph_name: str) -> dict[str, Any]:
        """Return a retained Phase 4 graph without exposing arbitrary paths."""
        graphs = {
            "ring01": self.phase4_ring01_graph,
            "synthetic-ten-stone": self.phase4_synthetic_graph,
        }
        graph = graphs.get(graph_name)
        if graph is None:
            raise ObjectNotFoundError(f"Unknown Phase 4 graph: {graph_name}")
        return deepcopy(graph)

    def _reload_phase4_candidate_review(self) -> dict[str, Any]:
        _, self.phase4_candidate_review = load_phase4_candidate_review(
            self.repo_root
        )
        return self.phase4_candidate_review

    def _phase4_candidate_review_summary(self) -> dict[str, Any]:
        manifest = self.phase4_candidate_review
        return {
            "schema_version": manifest.get("schema_version"),
            "authority_policy": manifest.get("authority_policy"),
            "semantic_labels": deepcopy(manifest.get("semantic_labels", {})),
            "review_contract": deepcopy(manifest.get("review_contract", {})),
            "gate": deepcopy(manifest.get("gate", {})),
            "objects": [
                {
                    "object_id": object_id,
                    "split": record.get("split"),
                    "pending_candidate_count": record.get(
                        "pending_candidate_count", 0
                    ),
                    "semantic_review_complete": bool(
                        record.get("semantic_review_complete", False)
                    ),
                }
                for object_id, record in manifest.get("objects", {}).items()
            ],
        }

    def phase4_candidate_review_summary(self) -> dict[str, Any]:
        self._reload_phase4_candidate_review()
        return self._phase4_candidate_review_summary()

    def phase4_candidate_review_object(self, object_id: str) -> dict[str, Any]:
        self._record(object_id)
        manifest = self._reload_phase4_candidate_review()
        try:
            record = manifest["objects"][object_id]
        except KeyError as exc:
            raise ObjectNotFoundError(
                f"No Phase 4 candidate review record for {object_id}"
            ) from exc
        return {
            "schema_version": manifest.get("schema_version"),
            "authority_policy": manifest.get("authority_policy"),
            "semantic_labels": deepcopy(manifest.get("semantic_labels", {})),
            "dataset_gate": deepcopy(manifest.get("gate", {})),
            "object": deepcopy(record),
        }

    def phase4_candidate_asset(
        self,
        object_id: str,
        view: str,
        kind: str,
        candidate_id: str | None = None,
    ) -> Path:
        self._record(object_id)
        manifest = self._reload_phase4_candidate_review()
        if view not in VIEW_ORDER:
            raise AssetNotFoundError(f"Unknown Phase 4 candidate view: {view}")
        try:
            view_record = manifest["objects"][object_id]["views"][view]
        except KeyError as exc:
            raise AssetNotFoundError(
                f"Candidate evidence unavailable for {object_id}:{view}"
            ) from exc
        if kind == "source":
            return self._safe_path(
                self.prepared_root / "images" / object_id / f"{view}.png"
            )
        if kind == "overlay":
            return self._safe_path(view_record["overlay_path"])
        if kind != "mask":
            raise AssetNotFoundError(f"Unknown Phase 4 candidate asset: {kind}")
        matches = [
            item
            for item in view_record["candidates"]
            if item["candidate_id"] == candidate_id
        ]
        if len(matches) != 1:
            raise AssetNotFoundError(
                f"Unknown candidate ID for {object_id}:{view}: {candidate_id}"
            )
        return self._safe_path(matches[0]["mask_path"])

    def review_phase4_candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_phase4_candidate_review(
                self.repo_root,
                str(payload.get("object_id", "")),
                str(payload.get("view", "")),
                str(payload.get("candidate_id", "")),
                str(payload.get("semantic_label", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("note"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.phase4_candidate_review = manifest
        return self.phase4_candidate_review_object(
            str(payload.get("object_id", ""))
        )

    def preview_phase4_edit(
        self, graph_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Preview a graph-state edit in memory; never persist or generate CAD."""
        component_id = str(payload.get("component_id", "")).strip()
        operation = str(payload.get("operation", "")).strip()
        parameters = payload.get("parameters", {})
        if not component_id:
            raise DatasetError("Component ID is required")
        if not isinstance(parameters, dict):
            raise DatasetError("Edit parameters must be a JSON object")
        try:
            edited, audit = apply_local_edit(
                self.phase4_graph(graph_name),
                component_id,
                operation,
                parameters,
            )
        except ComponentGraphError as exc:
            raise DatasetError(str(exc)) from exc
        return {
            "dry_run": True,
            "graph_state_only": True,
            "geometry_generated": False,
            "audit": audit,
            "graph": edited,
        }

    def capabilities(self) -> dict[str, Any]:
        """Return the working Studio feature set with evidence-backed coverage."""

        complete = self.phase1_complete
        complete_coverage = complete.get("coverage", {})
        complete_contract = complete.get("feature_contract", {})
        roundtrip = complete.get("reconstruction_contract", {})
        metric = self.metric_depth_consensus
        metric_coverage = metric.get("backend_coverage", {})
        phase3_objects = list(self._phase3_objects.values())
        geometry = self.phase3_geometry
        geometry_colmap = geometry.get("colmap", {})
        geometry_backends = geometry.get("backends", {})
        machine_claims = self.phase2_machine_review.get("claims", {})
        stl1_gate = self._stl1_phase1_review_summary().get("gate", {})

        cards = [
            {
                "id": "reproducibility",
                "title": "Reproducible software baseline",
                "phase": "Bible Phase 0",
                "status": "complete",
                "coverage": self.phase0_environment.get("regression", {}).get(
                    "result", "unknown"
                ),
                "summary": "Pinned environment, dependency inventory and automated regression evidence are retained.",
                "functions": [
                    "Inspect the environment and dependency manifests",
                    "Re-run the repository validation suite in the pinned container",
                    "Keep model and dataset claims tied to versioned evidence",
                ],
                "limitations": deepcopy(
                    self.phase0_environment.get("known_limits", [])[:2]
                ),
                "actions": [
                    {"label": "Environment", "href": "/api/artifacts/phase0_environment"},
                    {"label": "Gate report", "href": "/api/artifacts/phase0_status"},
                ],
            },
            {
                "id": "intake_and_audit",
                "title": "Single and five-view intake auditor",
                "phase": "Image intake",
                "status": "implemented",
                "coverage": "1 image or 5 named views",
                "summary": "The reusable local auditor validates inputs and produces normalized, segmentation, feature and phase-audit evidence without modifying source files.",
                "functions": [
                    "Reject corrupt or incomplete view sets",
                    "Retain source and copied-image hashes",
                    "Generate a self-contained audit report and comparison sheet",
                ],
                "limitations": [
                    "The auditor currently runs as a local command, outside the Flask request lifecycle."
                ],
                "actions": [
                    {"label": "Usage", "href": "/api/artifacts/upload_auditor_doc"}
                ],
            },
            {
                "id": "feature_extraction",
                "title": "Complete image feature extraction",
                "phase": "Image-only Phase 1",
                "status": "implemented",
                "coverage": (
                    f"{int(complete_coverage.get('view_count', 0))} views · "
                    f"{int(complete_contract.get('stored_array_count_per_view', 0))} arrays/view"
                ),
                "summary": "Normalized pixels, silhouettes, negative-space proposals, edges, colour matrices and differential features are retained per view.",
                "functions": [
                    "Inspect reference, normalized, mask and edge layers",
                    "Compare per-view contours, holes, occupancy and symmetry",
                    "Reconstruct retained source pixels exactly",
                ],
                "limitations": [
                    "Semantic feature matrices alone are not an invertible replacement for retained RGB pixels."
                ],
                "checks": {
                    "all_views_processed": int(complete_coverage.get("view_count", 0))
                    == len(self.records) * len(VIEW_ORDER),
                    "source_pixel_roundtrip_exact": bool(
                        roundtrip.get("source_pixel_roundtrip_exact", False)
                    ),
                },
                "actions": [
                    {"label": "Open object evidence", "href": "#object-workspace"},
                    {"label": "Feature report", "href": "/api/artifacts/dataset_complete_phase1_report"},
                ],
            },
            {
                "id": "human_review",
                "title": "Evidence review and mask correction",
                "phase": "Image-only Phase 1",
                "status": "implemented",
                "coverage": f"{int(stl1_gate.get('evidence_review_count', 0))} review decisions",
                "summary": "Ring01 and STL-1 review controls record identified decisions without overwriting machine proposals.",
                "functions": [
                    "Approve, reject or flag masks, edges and depth evidence",
                    "Record cross-view stones, prongs and relief inventory",
                    "Upload a corrected binary silhouette as a separate artifact",
                ],
                "limitations": [
                    f"{int(stl1_gate.get('pending_evidence_count', 0))} STL-1 evidence decisions remain pending."
                ],
                "actions": [
                    {"label": "Ring01 review", "href": "#phase1-title"},
                    {"label": "STL-1 review", "href": "#object-workspace"},
                ],
            },
            {
                "id": "depth_evidence",
                "title": "Relative and multi-model depth evidence",
                "phase": "Image-only Phase 1/2",
                "status": "partial",
                "coverage": (
                    f"Depth Anything V2: {int(self.phase1_depth.get('coverage', {}).get('view_count', 0))} views · "
                    f"consensus: {int(metric.get('sample_count', 0))} samples"
                ),
                "summary": "Depth Anything V2, UniDepth V2 and Metric3D V2 have complete selected-sample coverage; Depth Pro has representative coverage and disagreement maps are retained.",
                "functions": [
                    "Inspect relative depth, flip ensemble and uncertainty layers",
                    "Compare same-view predictions across four depth backends",
                    "Identify stable and appearance-sensitive depth regions",
                ],
                "limitations": [
                    f"Depth Pro coverage is {int(metric_coverage.get('depth_pro', {}).get('sample_count', 0))}/{int(metric_coverage.get('depth_pro', {}).get('requested_sample_count', 0))} because of host-memory limits.",
                    "Cross-view metric alignment is incomplete; model agreement is not physical-depth accuracy.",
                ],
                "actions": [
                    {"label": "Depth evidence", "href": "#metric-depth-evidence"},
                    {"label": "Consensus report", "href": "/api/artifacts/metric_depth_consensus_report"},
                ],
            },
            {
                "id": "reflection_resilience",
                "title": "Reflection-sensitive silhouette correction",
                "phase": "Image-only Phase 2",
                "status": "partial",
                "coverage": "Ring01 angled view",
                "summary": "A topology-aware correction prevents bright inner-shank reflections from being treated as metal foreground.",
                "functions": [
                    "Compare source, legacy mask, corrected mask and uncertainty",
                    "Preserve the baseline proposal for auditability",
                    "Carry the correction into machine-reviewed image-only research",
                ],
                "limitations": [
                    "The correction is machine-validated but still awaits identified human approval."
                ],
                "checks": {
                    "machine_evidence_validated": bool(
                        machine_claims.get("machine_evidence_validated", False)
                    ),
                    "human_review_performed": bool(
                        machine_claims.get("human_review_performed", False)
                    ),
                },
                "actions": [
                    {"label": "Inspect correction", "href": "#phase2-paired-title"},
                    {"label": "Machine report", "href": "/api/artifacts/phase2_machine_review"},
                ],
            },
            {
                "id": "generalisation",
                "title": "External real-image generalisation audit",
                "phase": "Phase 1/2 validation",
                "status": "partial",
                "coverage": f"{int(self.external_ring_report.get('sample_count', 0))} external images",
                "summary": "OrnAsia and Segment Your Ring inputs exercise preprocessing and segmentation outside STL-1.",
                "functions": [
                    "Audit segmentation on complex real-image backgrounds",
                    "Compare ten predictions with supplied whole-ring masks",
                    "Track source licences and review state",
                ],
                "limitations": [
                    f"{int(self.external_ring_report.get('review_required_mask_count', 0))} masks remain review-gated.",
                    "The sources contain no camera, scale, component or CAD truth.",
                ],
                "actions": [
                    {"label": "Open generalisation", "href": "#external-ring-title"},
                    {"label": "Intake report", "href": "/api/artifacts/external_ring_report"},
                ],
            },
            {
                "id": "multiview_geometry",
                "title": "Experimental multi-view geometry",
                "phase": "Geometry evidence",
                "status": "partial",
                "coverage": f"{len(phase3_objects)} coarse meshes · {int(geometry.get('selected_object_count', 0))} COLMAP probes · {int(self.vggt_multiring_validation.get('object_count', 0))} VGGT ring tests",
                "summary": "Visual-hull research meshes and COLMAP probes cover STL-1; VGGT dense geometry now includes one training ring and two held-out test rings.",
                "functions": [
                    "Inspect five-view reprojection and research-mesh previews",
                    "Download watertight experimental STL and 3MF meshes",
                    "Inspect COLMAP feature and match evidence per object",
                ],
                "limitations": [
                    f"COLMAP produced {int(geometry_colmap.get('completed_sparse_model_count', 0))} verified sparse models.",
                    f"VGGT accepted {int(self.vggt_multiring_validation.get('accepted_object_count', 0))} of {int(self.vggt_multiring_validation.get('object_count', 0))} tested STL-1 objects at the cross-view geometry gate.",
                    "External real-image sets are single-view or identity-unassigned and cannot be treated as multi-view geometry inputs.",
                ],
                "checks": {
                    "research_meshes_watertight": bool(
                        self.phase3_report.get("all_meshes_watertight", False)
                    ),
                    "cross_view_metric_alignment": False,
                },
                "actions": [
                    {"label": "Open experimental 3D", "href": "#object-workspace"},
                    {"label": "Geometry report", "href": "/api/artifacts/phase3_geometry_report"},
                ],
            },
            {
                "id": "component_graph",
                "title": "Component and constraint graph",
                "phase": "Bible Phase 4",
                "status": "partial",
                "coverage": f"{int(self.phase4_component_graph.get('ring01', {}).get('node_count', 0))} Ring01 proposal nodes · {int(self.phase4_prong_instances.get('summary', {}).get('resolved_instance_view_count', 0))}/5 prong-instance views resolved · {int(self.phase4_stl1_candidates.get('candidate_count', 0))} STL-1 negative-space candidates · 30 synthetic edit cases",
                "summary": "Stable-ID graph API, dependencies and local-edit previews are implemented. VGGT supports the resolved Ring01 prong ordering, while a dataset-wide STL-1 inventory retains apertures and openwork without falsely promoting them to stones.",
                "functions": [
                    "Inspect Ring01 machine-proposal component dependencies",
                    "Compare individual prong proposals with unresolved projection groups",
                    "Inspect VGGT component-centroid assignment matrices and confidence",
                    "Inspect dataset-wide enclosed negative-space candidates",
                    "Preview remove, resize and replace transactions without persistence",
                    "Verify stable IDs and unchanged unrelated graph state",
                ],
                "limitations": [
                    "Ring01 component identities remain pending human review.",
                    "Side and back images do not individually resolve overlapping prong pairs.",
                    "VGGT agreement is non-metric machine evidence, not identity ground truth.",
                    "STL-1 is monochrome and unlabelled, so aperture evidence cannot confirm gemstone material or cross-view component identity.",
                    "The ten-stone locality result covers graph state, not CAD Boolean geometry.",
                ],
                "checks": deepcopy(self.phase4_component_graph.get("checks", {})),
                "actions": [
                    {"label": "Graph report", "href": "/api/artifacts/phase4_component_graph_report"},
                    {"label": "Prong audit", "href": "/api/artifacts/phase4_prong_instance_audit"},
                    {"label": "Track audit", "href": "/api/artifacts/phase4_component_track_audit"},
                    {"label": "STL-1 candidate audit", "href": "/api/artifacts/phase4_stl1_candidate_audit"},
                    {"label": "Ring01 graph", "href": "/api/phase4/graphs/ring01"},
                ],
            },
            {
                "id": "parametric_cad",
                "title": "Editable structural CAD",
                "phase": "Bible Phase 4+",
                "status": "blocked",
                "coverage": "Not implemented",
                "summary": "The component graph foundation exists, but independently regenerated stones, prongs, seats, cavities and exact STEP construction remain future work.",
                "functions": [],
                "limitations": [
                    "Current STL and 3MF outputs are research meshes, not authoritative editable CAD.",
                    "Image-only inputs do not verify absolute millimetres without a known scale reference.",
                ],
                "actions": [],
            },
        ]

        remaining_gates = [
            {
                "id": "cross_view_alignment",
                "title": "Cross-view camera and depth alignment",
                "status": "next",
                "evidence": "Same-view depth consensus exists; COLMAP did not form verified sparse models.",
                "requirement": "Evaluate MASt3R/VGGT within hardware and licence limits, then fuse only geometrically consistent pixels.",
            },
            {
                "id": "component_graph",
                "title": "Universal jewellery component graph",
                "status": "partial_review_blocked",
                "evidence": "Stable-ID API and 30-case synthetic locality benchmark exist; Ring01 graph has 10 pending proposal nodes.",
                "requirement": "Approve or correct image-space component identities, add individual instance masks, then validate local CAD regeneration.",
            },
            {
                "id": "exact_cad",
                "title": "Parametric CAD reconstruction",
                "status": "not_implemented",
                "evidence": "Coarse watertight STL/3MF research meshes exist.",
                "requirement": "Fit editable dimensions and topology, generate exact B-rep/STEP, and validate renders against every supplied view.",
            },
            {
                "id": "image_only_scale",
                "title": "Absolute scale and manufacturing claims",
                "status": "unavailable_from_images_alone",
                "evidence": "Relative geometry and uncertainty can be estimated.",
                "requirement": "Keep dimensions scale-normalized unless an image contains a reliable known dimension; do not claim verified millimetres otherwise.",
            },
        ]

        status_counts = {
            status: sum(card["status"] == status for card in cards)
            for status in ("complete", "implemented", "partial", "blocked")
        }
        return {
            "schema_version": "studio_capabilities_v1",
            "execution_profile": "images_only",
            "source_of_truth": "checked_in_reports_and_runtime_contracts",
            "summary": {
                "capability_count": len(cards),
                "working_count": status_counts["complete"]
                + status_counts["implemented"],
                "partial_count": status_counts["partial"],
                "blocked_count": status_counts["blocked"],
                "status_counts": status_counts,
            },
            "capabilities": cards,
            "remaining_gates": remaining_gates,
            "claim_boundary": "High-fidelity image and structural evidence is supported; exact metric or manufacturing accuracy is not validated from the available images alone.",
        }

    def _reload_phase1_ground_truth(self) -> dict[str, Any]:
        self.phase1_ground_truth = self._read_json(
            self.repo_root / "data" / "ring01_ground_truth_v1" / "manifest.json"
        )
        return self.phase1_ground_truth

    def phase1_review_manifest(self) -> dict[str, Any]:
        return deepcopy(self._reload_phase1_ground_truth())

    def review_phase1_label(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_label_review(
                self.repo_root,
                str(payload.get("view", "")),
                str(payload.get("label", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("note"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.phase1_ground_truth = manifest
        return self._phase1_ground_truth_summary()

    def review_phase1_identity(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_identity_review(
                self.repo_root,
                str(payload.get("component_id", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("note"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.phase1_ground_truth = manifest
        return self._phase1_ground_truth_summary()

    def review_phase1_camera(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_camera_review(
                self.repo_root,
                str(payload.get("view", "")),
                str(payload.get("projection_model", "unknown")),
                str(payload.get("uncertainty", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("note"),
                payload.get("intrinsics"),
                payload.get("extrinsics"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.phase1_ground_truth = manifest
        return self._phase1_ground_truth_summary()

    def phase1_review_asset(
        self, view: str, kind: str, label: str | None = None
    ) -> Path:
        manifest = self._reload_phase1_ground_truth()
        if view not in PHASE1_VIEWS:
            raise AssetNotFoundError(f"Unknown Phase 1 view: {view}")
        record = manifest["views"][view]
        if kind == "source":
            candidate = record["source"]["path"]
        elif kind == "proposal":
            if label not in PHASE1_LABELS:
                raise AssetNotFoundError(f"Unknown Phase 1 label: {label}")
            candidate = record["proposals"][label]["path"]
        elif kind == "review_sheet":
            candidate = record["review_sheet"]
        else:
            raise AssetNotFoundError(f"Unknown Phase 1 review asset: {kind}")
        return self._safe_path(self.repo_root / candidate)

    def _phase1_ground_truth_summary(self) -> dict[str, Any]:
        manifest = self.phase1_ground_truth
        gate = manifest.get("exit_gate", {})
        depth = manifest.get("depth_anything_v2", {})
        return {
            "object_id": manifest.get("object_id"),
            "status": manifest.get("status"),
            "exit_gate": {
                "status": gate.get("status"),
                "pending_label_review_count": len(
                    gate.get("pending_label_reviews", [])
                ),
                "pending_component_identity_count": len(
                    gate.get("pending_component_identities", [])
                ),
                "pending_camera_record_count": len(
                    gate.get("pending_camera_records", [])
                ),
                "physical_scale_missing": bool(
                    gate.get("physical_scale_missing", True)
                ),
                "physical_scale_required": bool(
                    gate.get("physical_scale_required", True)
                ),
                "metric_scale_available": bool(
                    gate.get("metric_scale_available", False)
                ),
                "image_ground_truth_complete": bool(
                    gate.get("image_ground_truth_complete", False)
                ),
                "human_ground_truth_complete": bool(
                    gate.get("human_ground_truth_complete", False)
                ),
            },
            "component_count": len(manifest.get("component_catalog", [])),
            "physical_scale": deepcopy(manifest.get("physical_scale", {})),
            "depth": deepcopy(depth),
            "depth_validation": {
                "status": self.phase1_depth_validation.get("status"),
                "checks": deepcopy(self.phase1_depth_validation.get("checks", {})),
                "runtime": deepcopy(self.phase1_depth_validation.get("runtime", {})),
                "metric_depth": deepcopy(
                    self.phase1_depth_validation.get("metric_depth", {})
                ),
                "decision": self.phase1_depth_validation.get("decision"),
                "front_raw_object_top_bottom_delta": self.phase1_depth_validation
                .get("views", {})
                .get("front", {})
                .get("whole_object_distribution", {})
                .get("top_minus_bottom_mean"),
                "front_regularized_object_top_bottom_delta": self.phase1_depth_validation
                .get("views", {})
                .get("front", {})
                .get("regularized_whole_object_distribution", {})
                .get("top_minus_bottom_mean"),
                "front_raw_stone_top_bottom_delta": self.phase1_depth_validation
                .get("views", {})
                .get("front", {})
                .get("stone_distribution", {})
                .get("top_minus_bottom_mean"),
                "front_regularized_stone_top_bottom_delta": self.phase1_depth_validation
                .get("views", {})
                .get("front", {})
                .get("regularized_stone_distribution", {})
                .get("top_minus_bottom_mean"),
            },
            "pixel_features": {
                "status": self.phase1_pixel_features.get("status"),
                "view_count": self.phase1_pixel_features.get("view_count"),
                "feature_groups": deepcopy(
                    self.phase1_pixel_features.get("feature_groups", {})
                ),
                "reconstruction_contract": deepcopy(
                    self.phase1_pixel_features.get("reconstruction_contract", {})
                ),
                "depth_contract": deepcopy(
                    self.phase1_pixel_features.get("depth_contract", {})
                ),
                "authority": self.phase1_pixel_features.get("authority"),
            },
            "authority_policy": manifest.get("authority_policy"),
        }

    def _reload_stl1_phase1_review(self) -> dict[str, Any]:
        _, self.stl1_phase1_review = load_stl1_phase1_manifest(self.repo_root)
        return self.stl1_phase1_review

    def _stl1_phase1_review_summary(self) -> dict[str, Any]:
        manifest = self.stl1_phase1_review
        objects = []
        for object_id, record in manifest.get("objects", {}).items():
            gate = record.get("gate", {})
            objects.append(
                {
                    "object_id": object_id,
                    "source_name": record.get("source_name", object_id),
                    "split": record.get("split"),
                    "status": gate.get("status"),
                    "pending_evidence_count": gate.get(
                        "pending_evidence_count", 0
                    ),
                    "object_review_pending": bool(
                        gate.get("object_review_pending", True)
                    ),
                }
            )
        return {
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
            "authority_policy": manifest.get("authority_policy"),
            "gate": deepcopy(manifest.get("gate", {})),
            "evidence_types": deepcopy(manifest.get("evidence_types", {})),
            "review_contract": deepcopy(manifest.get("review_contract", {})),
            "automated_checks": deepcopy(manifest.get("automated_checks", {})),
            "limitations": deepcopy(manifest.get("limitations", [])),
            "objects": objects,
        }

    def stl1_phase1_review_summary(self) -> dict[str, Any]:
        self._reload_stl1_phase1_review()
        return self._stl1_phase1_review_summary()

    def stl1_phase1_review_object(self, object_id: str) -> dict[str, Any]:
        self._record(object_id)
        manifest = self._reload_stl1_phase1_review()
        try:
            record = manifest["objects"][object_id]
        except KeyError as exc:
            raise ObjectNotFoundError(
                f"No STL-1 Phase 1 review record for {object_id}"
            ) from exc
        return {
            "schema_version": manifest.get("schema_version"),
            "authority_policy": manifest.get("authority_policy"),
            "evidence_types": deepcopy(manifest.get("evidence_types", {})),
            "dataset_gate": deepcopy(manifest.get("gate", {})),
            "object": deepcopy(record),
        }

    def review_stl1_phase1_evidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_stl1_evidence_review(
                self.repo_root,
                str(payload.get("object_id", "")),
                str(payload.get("view", "")),
                str(payload.get("evidence_type", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("note"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.stl1_phase1_review = manifest
        return self.stl1_phase1_review_object(str(payload.get("object_id", "")))

    def review_stl1_phase1_object(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_stl1_object_review(
                self.repo_root,
                str(payload.get("object_id", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("component_inventory", {}),
                payload.get("note"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.stl1_phase1_review = manifest
        return self.stl1_phase1_review_object(str(payload.get("object_id", "")))

    def review_stl1_phase1_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest = apply_stl1_view_review(
                self.repo_root,
                str(payload.get("object_id", "")),
                str(payload.get("view", "")),
                str(payload.get("decision", "")),
                str(payload.get("reviewer", "")),
                payload.get("note"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(str(exc)) from exc
        self.stl1_phase1_review = manifest
        return self.stl1_phase1_review_object(str(payload.get("object_id", "")))

    def correct_stl1_silhouette(
        self,
        object_id: str,
        view: str,
        image_bytes: bytes,
        reviewer: str,
        note: str | None,
    ) -> dict[str, Any]:
        try:
            manifest = save_corrected_silhouette(
                self.repo_root,
                object_id,
                view,
                image_bytes,
                reviewer,
                note,
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise DatasetError(str(exc)) from exc
        self.stl1_phase1_review = manifest
        return self.stl1_phase1_review_object(object_id)

    def stl1_phase1_review_asset(
        self,
        object_id: str,
        view: str,
        evidence_type: str,
        variant: str = "active",
    ) -> Path:
        self._record(object_id)
        manifest = self._reload_stl1_phase1_review()
        if view not in STL1_PHASE1_VIEWS:
            raise AssetNotFoundError(f"Unknown STL-1 Phase 1 view: {view}")
        if evidence_type not in STL1_PHASE1_EVIDENCE_TYPES:
            raise AssetNotFoundError(
                f"Unknown STL-1 Phase 1 evidence type: {evidence_type}"
            )
        if variant not in {"proposal", "corrected", "active"}:
            raise AssetNotFoundError(f"Unknown review asset variant: {variant}")
        review = manifest["objects"][object_id]["views"][view]["evidence"][
            evidence_type
        ]
        key = f"{variant}_path"
        candidate = review.get(key)
        if not candidate:
            raise AssetNotFoundError(
                f"{variant} {evidence_type} is unavailable for {object_id}:{view}"
            )
        return self._safe_path(candidate)

    def list_objects(self, split: str | None = None) -> list[dict[str, Any]]:
        if split not in {None, "all", "train", "val", "test"}:
            raise DatasetError(f"Invalid split: {split}")
        result = []
        for record in self.records:
            if split not in {None, "all"} and record["split"] != split:
                continue
            phase3 = self._phase3_objects.get(record["object_id"], {})
            phase1_review = self.stl1_phase1_review.get("objects", {}).get(
                record["object_id"], {}
            )
            result.append(
                {
                    "object_id": record["object_id"],
                    "source_name": record.get("source_object_name", record["object_id"]),
                    "category": record.get("category", "unknown"),
                    "split": record["split"],
                    "view_count": len(record["views"]),
                    "phase3_mean_iou": phase3.get("reprojection", {}).get("mean_iou"),
                    "phase3_available": bool(phase3),
                    "phase1_review_status": phase1_review.get("gate", {}).get(
                        "status", "unavailable"
                    ),
                    "phase1_pending_review_count": phase1_review.get("gate", {}).get(
                        "pending_evidence_count", 0
                    )
                    + int(
                        phase1_review.get("gate", {}).get(
                            "object_review_pending", False
                        )
                    ),
                }
            )
        return result

    def object_detail(self, object_id: str) -> dict[str, Any]:
        record = self._record(object_id)
        phase1_review = deepcopy(
            self.stl1_phase1_review.get("objects", {}).get(object_id, {})
        )
        phase1 = self._phase1_by_object.get(object_id, {})
        phase1_depth = self._phase1_depth_by_object.get(object_id, {})
        phase1_complete = self._phase1_complete_by_object.get(object_id, {})
        phase2 = self._phase2_metrics.get(object_id, {})
        phase3 = deepcopy(self._phase3_objects.get(object_id))
        geometry_evidence = deepcopy(self._phase3_geometry_objects.get(object_id))
        views: dict[str, Any] = {}
        for view in VIEW_ORDER:
            metadata = record["views"][view]
            complete_view = phase1_complete.get("views", {}).get(view, {})
            views[view] = {
                "label": VIEW_LABELS[view],
                "source_size_wh": deepcopy(metadata.get("source_size_wh")),
                "prepared_size_wh": deepcopy(metadata.get("prepared_size_wh")),
                "mask_occupancy": metadata.get("mask_occupancy"),
                "camera_calibrated": bool(metadata.get("camera_calibrated", False)),
                "source_sha256": metadata.get("source_sha256"),
                "features": deepcopy(phase1.get(view, {})),
                "depth": deepcopy(phase1_depth.get("views", {}).get(view, {})),
                "complete_features": {
                    "feature_count": complete_view.get("feature_count"),
                    "repeat_count": complete_view.get("repeat_count"),
                    "repeat_deterministic": complete_view.get(
                        "repeat_deterministic"
                    ),
                    "source_pixel_roundtrip_exact": complete_view.get(
                        "source_pixel_roundtrip_exact"
                    ),
                    "horizontal_flip_error_over_depth_span": complete_view.get(
                        "horizontal_flip_error_over_depth_span"
                    ),
                    "vertical_flip_error_over_depth_span": complete_view.get(
                        "vertical_flip_error_over_depth_span"
                    ),
                    "raw_distribution": deepcopy(
                        complete_view.get("raw_distribution", {})
                    ),
                    "regularized_distribution": deepcopy(
                        complete_view.get("regularized_distribution", {})
                    ),
                    "metric": bool(complete_view.get("metric", False)),
                    "cross_view_aligned": bool(
                        complete_view.get("cross_view_aligned", False)
                    ),
                },
                "metrics": deepcopy(phase2.get(view)),
            }
        return {
            "object_id": object_id,
            "source_name": record.get("source_object_name", object_id),
            "category": record.get("category", "unknown"),
            "split": record["split"],
            "views": views,
            "view_order": list(VIEW_ORDER),
            "phase3": phase3,
            "geometry_evidence": geometry_evidence,
            "complete_phase1": {
                "status": phase1_complete.get("status"),
                "audit_sheet": phase1_complete.get("audit_sheet"),
                "lsv_rsv_depth_distribution_difference": phase1_complete.get(
                    "lsv_rsv_depth_distribution_difference"
                ),
            },
            "phase1_review": {
                "gate": deepcopy(phase1_review.get("gate", {})),
                "object_review": deepcopy(
                    phase1_review.get("object_review", {})
                ),
            },
            "supervision": deepcopy(record.get("supervision", {})),
            "bible": {
                "capture": {
                    "tier": "A",
                    "view_contract": "five explicitly named research views",
                    "allowed_claims": [
                        "segmentation proposals",
                        "visual proposals",
                        "coarse non-metric reconstruction",
                    ],
                    "forbidden_claim": "manufacturing-accurate geometry",
                },
                "evidence_records": [
                    {
                        "stage": "immutable source views",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "observed",
                    },
                    {
                        "stage": "normalization, masks and edges",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "inferred",
                    },
                    {
                        "stage": "foreground, edge, contour, hole and symmetry measurements",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "measured_non_metric",
                    },
                    {
                        "stage": "Depth Anything V2 relative-depth and uncertainty proposals",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "inferred_non_metric",
                    },
                    {
                        "stage": "22-matrix reversible pixel evidence and three-run depth validation",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "measured_non_metric",
                    },
                    {
                        "stage": "visual-hull mesh",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "inferred",
                    },
                    {
                        "stage": "COLMAP camera and sparse-geometry baseline",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "measured_no_verified_cross_view_matches",
                    },
                    {
                        "stage": "VGGT camera and dense-geometry adapter",
                        "evidence_class": "PROVEN-IN-PROJECT",
                        "state": "blocked_missing_approved_checkpoint_and_hardware_gate",
                    },
                ],
                "gates": {
                    "human_reviewed_image_truth": bool(
                        phase1_review.get("gate", {}).get(
                            "human_review_complete", False
                        )
                    ),
                    "metric_scale": False,
                    "calibrated_cameras": False,
                    "reviewed_component_graph": False,
                    "authoritative_exact_brep": False,
                    "manufacturing_validation": False,
                },
                "artifact_authority": "No authoritative STEP exists for this dataset object; STL and 3MF are derived research artifacts.",
                "decision": "research_only",
            },
            "artifacts": self.artifact_availability(object_id),
        }

    def artifact_availability(self, object_id: str) -> dict[str, Any]:
        self._record(object_id)
        view_assets = {
            view: {
                kind: self._asset_exists(object_id, kind, view)
                for kind in (
                    "source",
                    "normalized",
                    "mask",
                    "edges",
                    "depth",
                    "depth_uncertainty",
                    "depth_ensemble",
                    "depth_transform_uncertainty",
                    "depth_overlay",
                    "view_audit",
                )
            }
            for view in VIEW_ORDER
        }
        return {
            "views": view_assets,
            "phase_sheet": self._asset_exists(object_id, "phase_sheet"),
            "phase1_depth_review": self._asset_exists(
                object_id, "phase1_depth_review"
            ),
            "phase1_complete_audit": self._asset_exists(
                object_id, "phase1_complete_audit"
            ),
            "phase3_preview": self._asset_exists(object_id, "phase3_preview"),
            "phase3_stl": self._asset_exists(object_id, "phase3_stl"),
            "phase3_3mf": self._asset_exists(object_id, "phase3_3mf"),
            "colmap_report": self._asset_exists(object_id, "colmap_report"),
            "held_out_review": self._asset_exists(object_id, "held_out_review"),
        }

    def _asset_exists(self, object_id: str, kind: str, view: str | None = None) -> bool:
        try:
            self.asset_path(object_id, kind, view)
        except AssetNotFoundError:
            return False
        return True

    def asset_path(self, object_id: str, kind: str, view: str | None = None) -> Path:
        record = self._record(object_id)
        view_kinds = {
            "source": "source_path",
            "normalized": "image_path",
            "mask": "jewelry_mask_path",
            "edges": "edge_path",
        }
        if kind in view_kinds:
            if view not in VIEW_ORDER:
                raise AssetNotFoundError(f"A valid view is required for {kind}")
            return self._safe_path(record["views"][view][view_kinds[kind]])
        if kind in {"depth", "depth_uncertainty"}:
            if view not in VIEW_ORDER:
                raise AssetNotFoundError(f"A valid view is required for {kind}")
            depth = self._phase1_depth_by_object.get(object_id, {})
            field = "depth_u16" if kind == "depth" else "uncertainty_u16"
            path = depth.get("views", {}).get(view, {}).get(field)
            if not path:
                raise AssetNotFoundError(f"{kind} is unavailable for {object_id}:{view}")
            return self._safe_path(path)
        if kind in {
            "depth_ensemble",
            "depth_transform_uncertainty",
            "depth_overlay",
        }:
            if view not in VIEW_ORDER:
                raise AssetNotFoundError(f"A valid view is required for {kind}")
            complete = self._phase1_complete_by_object.get(object_id, {})
            fields = {
                "depth_ensemble": "regularized_depth_u16",
                "depth_transform_uncertainty": "uncertainty_u16",
                "depth_overlay": "photo_depth_overlay",
            }
            path = complete.get("views", {}).get(view, {}).get(fields[kind])
            if not path:
                raise AssetNotFoundError(
                    f"{kind} is unavailable for {object_id}:{view}"
                )
            return self._safe_path(path)
        if kind == "view_audit":
            if view not in VIEW_ORDER:
                raise AssetNotFoundError("A valid view is required for view_audit")
            return self._safe_path(
                self.phase_root
                / "phase_audits"
                / object_id
                / "views"
                / f"{object_id}_{view}_audit.png"
            )
        if kind == "phase_sheet":
            return self._safe_path(
                self.phase_root / "phase_audits" / object_id / f"{object_id}_all_phases.png"
            )
        if kind == "held_out_review":
            return self._safe_path(
                self.phase_root / "test_predictions" / f"{object_id}_review.png"
            )
        if kind == "phase1_depth_review":
            path = self._phase1_depth_by_object.get(object_id, {}).get("review_sheet")
            if not path:
                raise AssetNotFoundError(
                    f"Phase 1 depth review is unavailable for {object_id}"
                )
            return self._safe_path(path)
        if kind == "phase1_complete_audit":
            path = self._phase1_complete_by_object.get(object_id, {}).get(
                "audit_sheet"
            )
            if not path:
                raise AssetNotFoundError(
                    f"Complete Phase 1 audit is unavailable for {object_id}"
                )
            return self._safe_path(path)
        if kind == "colmap_report":
            geometry = self._phase3_geometry_objects.get(object_id, {})
            path = geometry.get("report")
            if not path:
                raise AssetNotFoundError(
                    f"COLMAP evidence is unavailable for {object_id}"
                )
            return self._safe_path(path)

        phase3 = self._phase3_objects.get(object_id)
        if not phase3:
            raise AssetNotFoundError(f"Phase 3 output is unavailable for {object_id}")
        export_keys = {
            "phase3_preview": "preview",
            "phase3_stl": "stl",
            "phase3_3mf": "3mf",
        }
        if kind in export_keys:
            path = phase3.get("exports", {}).get(export_keys[kind])
            if not path:
                raise AssetNotFoundError(f"{kind} is unavailable for {object_id}")
            return self._safe_path(path)
        raise AssetNotFoundError(f"Unknown artifact kind: {kind}")

    def global_asset_path(self, name: str) -> Path:
        allowed = {
            "multiview_overview": "dataset_multiview_overview.png",
            "processing_comparison": "dataset_processing_comparison.png",
            "held_out_comparison": "held_out_prediction_comparison.png",
        }
        reproducibility = {
            "phase0_environment": self.repo_root
            / "docs"
            / "bible_implementation"
            / "phase_0_reproducibility"
            / "environment_manifest.json",
            "phase0_licenses": self.repo_root
            / "docs"
            / "bible_implementation"
            / "phase_0_reproducibility"
            / "dependency_license_manifest.json",
            "phase0_status": self.repo_root
            / "docs"
            / "bible_implementation"
            / "phase_0_reproducibility"
            / "PHASE0_STATUS.md",
            "phase1_manifest": self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "manifest.json",
            "phase1_review": self.repo_root
            / self.phase1_ground_truth["review_summary"],
            "phase1_depth_review": self.repo_root
            / self.phase1_ground_truth["depth_anything_v2"]["review_summary"],
            "phase1_depth_metadata": self.repo_root
            / self.phase1_ground_truth["depth_anything_v2"]["metadata"],
            "dataset_depth_overview": self.repo_root
            / self.phase1_depth["overview"],
            "dataset_depth_report": self.phase_root
            / "phase1_depth_anything_v2"
            / "report.json",
            "dataset_complete_phase1_overview": self.repo_root
            / self.phase1_complete["overview"],
            "dataset_complete_phase1_report": self.phase_root
            / "phase1_complete_features_v1"
            / "report.json",
            "phase3_geometry_report": self.phase_root
            / "phase3_geometry_benchmark_v1"
            / "report.json",
            "phase4_component_graph_report": self.repo_root
            / "data"
            / "phase4_component_graph_v3"
            / "report.json",
            "phase4_ring01_graph": self.repo_root
            / "data"
            / "phase4_component_graph_v3"
            / "ring01_component_graph.json",
            "phase4_synthetic_graph": self.repo_root
            / "data"
            / "phase4_component_graph_v3"
            / "synthetic_ten_stone_graph.json",
            "phase4_prong_instance_report": self.repo_root
            / "data"
            / "phase4_prong_instances_v1"
            / "report.json",
            "phase4_prong_instance_audit": self.repo_root
            / "data"
            / "phase4_prong_instances_v1"
            / "ring01_prong_instance_audit.png",
            "phase4_component_track_report": self.repo_root
            / "data"
            / "phase4_component_tracks_v1"
            / "report.json",
            "phase4_component_track_audit": self.repo_root
            / "data"
            / "phase4_component_tracks_v1"
            / "ring01_component_track_audit.png",
            "phase4_stl1_candidate_report": self.repo_root
            / "data"
            / "phase4_stl1_component_candidates_v1"
            / "report.json",
            "phase4_stl1_candidate_audit": self.repo_root
            / "data"
            / "phase4_stl1_component_candidates_v1"
            / "stl1_front_component_candidate_audit.png",
            "cross_view_readiness_report": self.phase_root
            / "cross_view_backend_readiness_v1"
            / "report.json",
            "ring01_vggt_report": self.phase_root
            / "ring01_vggt_dense_224_v1"
            / "vggt_report.json",
            "ring01_vggt_validation": self.phase_root
            / "ring01_vggt_dense_224_v1"
            / "validation_report.json",
            "ring01_vggt_visual_review": self.phase_root
            / "ring01_vggt_dense_224_v1"
            / "machine_visual_review.json",
            "ring01_vggt_depth_preview": self.phase_root
            / "ring01_vggt_dense_224_v1"
            / "ring01_vggt_dense_review.png",
            "ring01_vggt_cloud_preview": self.phase_root
            / "ring01_vggt_dense_224_v1"
            / "ring01_vggt_fused_points_review.png",
            "ring01_vggt_point_cloud": self.phase_root
            / "ring01_vggt_dense_224_v1"
            / "ring01_vggt_fused_points.ply",
            "vggt_multiring_validation": self.phase_root
            / "vggt_multiring_224_v1"
            / "validation_report.json",
            "vggt_multiring_refinement_report": self.phase_root
            / "vggt_multiring_refinement_comparison_v1"
            / "report.json",
            "vggt_multiring_refinement_preview": self.phase_root
            / "vggt_multiring_refinement_comparison_v1"
            / "vggt_refinement_comparison.jpg",
            "semantic_camera_priors_report": self.phase_root
            / "semantic_camera_priors_v1"
            / "report.json",
            "semantic_camera_refinement_report": self.phase_root
            / "vggt_semantic_camera_refinement_v1"
            / "report.json",
            "vggt_view_transfer_report": self.phase_root
            / "vggt_view_transfer_v1"
            / "report.json",
            "vggt_view_transfer_ring_007": self.phase_root
            / "vggt_view_transfer_v1"
            / "ring_007"
            / "ring_007_view_transfer.png",
            "vggt_resolution_ladder_report": self.phase_root
            / "vggt_resolution_ladder_v1"
            / "report.json",
            "vggt_resolution_ladder_chart": self.phase_root
            / "vggt_resolution_ladder_v1"
            / "vggt_resolution_ladder.png",
            "vggt_resolution_visual_comparison": self.phase_root
            / "vggt_resolution_ladder_v1"
            / "vggt_224_vs_518_reprojection.jpg",
            "vggt_518_ring_001_reprojection": self.phase_root
            / "vggt_resolution_518_v1"
            / "ring_001"
            / "ring_001_cross_view_reprojection.png",
            "vggt_518_ring_007_reprojection": self.phase_root
            / "vggt_resolution_518_v1"
            / "ring_007"
            / "ring_007_cross_view_reprojection.png",
            "vggt_518_ring_023_reprojection": self.phase_root
            / "vggt_resolution_518_v1"
            / "ring_023"
            / "ring_023_cross_view_reprojection.png",
            "vggt_visual_hull_completion_report": self.phase_root
            / "vggt_visual_hull_completion_v1"
            / "report.json",
            "vggt_visual_hull_ring_004": self.phase_root
            / "vggt_visual_hull_completion_v1"
            / "ring_004_visual_hull_completion.png",
            "vggt_visual_hull_ring_020": self.phase_root
            / "vggt_visual_hull_completion_v1"
            / "ring_020_visual_hull_completion.png",
            "vggt_visual_hull_ring_007": self.phase_root
            / "vggt_visual_hull_completion_v1"
            / "ring_007_visual_hull_completion.png",
            "vggt_visual_hull_ring_015": self.phase_root
            / "vggt_visual_hull_completion_v1"
            / "ring_015_visual_hull_completion.png",
            "vggt_visual_hull_ring_023": self.phase_root
            / "vggt_visual_hull_completion_v1"
            / "ring_023_visual_hull_completion.png",
            "vggt_surface_aware_hull_report": self.phase_root
            / "vggt_surface_aware_hull_v1"
            / "report.json",
            "vggt_surface_hull_ring_004": self.phase_root
            / "vggt_surface_aware_hull_v1"
            / "ring_004_surface_aware_hull.png",
            "vggt_surface_hull_ring_020": self.phase_root
            / "vggt_surface_aware_hull_v1"
            / "ring_020_surface_aware_hull.png",
            "vggt_surface_hull_ring_007": self.phase_root
            / "vggt_surface_aware_hull_v1"
            / "ring_007_surface_aware_hull.png",
            "vggt_surface_hull_ring_015": self.phase_root
            / "vggt_surface_aware_hull_v1"
            / "ring_015_surface_aware_hull.png",
            "vggt_surface_hull_ring_023": self.phase_root
            / "vggt_surface_aware_hull_v1"
            / "ring_023_surface_aware_hull.png",
            "vggt_signed_depth_carving_report": self.phase_root
            / "vggt_signed_depth_carving_v1"
            / "report.json",
            "vggt_signed_depth_ring_004": self.phase_root
            / "vggt_signed_depth_carving_v1"
            / "ring_004_signed_depth_carving.png",
            "vggt_signed_depth_ring_020": self.phase_root
            / "vggt_signed_depth_carving_v1"
            / "ring_020_signed_depth_carving.png",
            "vggt_signed_depth_ring_007": self.phase_root
            / "vggt_signed_depth_carving_v1"
            / "ring_007_signed_depth_carving.png",
            "vggt_signed_depth_ring_015": self.phase_root
            / "vggt_signed_depth_carving_v1"
            / "ring_015_signed_depth_carving.png",
            "vggt_signed_depth_ring_023": self.phase_root
            / "vggt_signed_depth_carving_v1"
            / "ring_023_signed_depth_carving.png",
            "vggt_normal_consistency_report": self.phase_root
            / "vggt_normal_consistency_v1"
            / "report.json",
            "vggt_normal_consistency_ring_020": self.phase_root
            / "vggt_normal_consistency_v1"
            / "ring_020_normal_consistency.png",
            "vggt_normal_consistency_ring_015": self.phase_root
            / "vggt_normal_consistency_v1"
            / "ring_015_normal_consistency.png",
            "vggt_negative_space_report": self.phase_root
            / "vggt_negative_space_v1"
            / "report.json",
            "vggt_negative_space_ring_020": self.phase_root
            / "vggt_negative_space_v1"
            / "ring_020_negative_space.png",
            "vggt_negative_space_ring_015": self.phase_root
            / "vggt_negative_space_v1"
            / "ring_015_negative_space.png",
            "vggt_source_camera_alignment_report": self.phase_root
            / "vggt_source_camera_alignment_v1"
            / "report.json",
            "vggt_source_camera_alignment_ring_020": self.phase_root
            / "vggt_source_camera_alignment_v1"
            / "ring_020_source_camera_alignment.png",
            "vggt_source_camera_alignment_ring_015": self.phase_root
            / "vggt_source_camera_alignment_v1"
            / "ring_015_source_camera_alignment.png",
            "vggt_rigid_source_registration_report": self.phase_root
            / "vggt_rigid_source_registration_v1"
            / "report.json",
            "vggt_rigid_registration_ring_020_identity": self.phase_root
            / "vggt_rigid_source_registration_v1"
            / "ring_020_identity_registration_diagnostic.png",
            "vggt_rigid_registration_ring_020_medium": self.phase_root
            / "vggt_rigid_source_registration_v1"
            / "ring_020_rigid_medium_registration_diagnostic.png",
            "vggt_rigid_registration_ring_015": self.phase_root
            / "vggt_rigid_source_registration_v1"
            / "ring_015_rigid_source_registration.png",
            "vggt_track_evidence_report": self.phase_root
            / "vggt_track_evidence_v1"
            / "report.json",
            "vggt_track_guided_registration_report": self.phase_root
            / "vggt_track_guided_registration_v1"
            / "report.json",
            "vggt_tracks_ring_020": self.phase_root
            / "vggt_track_guided_registration_v1"
            / "ring_020_track_evidence.png",
            "vggt_tracks_ring_015": self.phase_root
            / "vggt_track_guided_registration_v1"
            / "ring_015_track_evidence.png",
            "vggt_track_registration_ring_020": self.phase_root
            / "vggt_track_guided_registration_v1"
            / "ring_020_track_guided_registration.png",
            "vggt_track_registration_ring_007": self.phase_root
            / "vggt_track_guided_registration_v1"
            / "ring_007_track_guided_registration.png",
            "vggt_track_registration_ring_015": self.phase_root
            / "vggt_track_guided_registration_v1"
            / "ring_015_track_guided_registration.png",
            "vggt_dense_track_evidence_report": self.phase_root
            / "vggt_dense_track_evidence_v1"
            / "report.json",
            "vggt_dense_cycle_registration_report": self.phase_root
            / "vggt_dense_cycle_consistent_registration_v1"
            / "report.json",
            "vggt_dense_cycle_ring_020": self.phase_root
            / "vggt_dense_cycle_consistent_registration_v1"
            / "ring_020_cycle_registration.png",
            "vggt_dense_cycle_ring_015": self.phase_root
            / "vggt_dense_cycle_consistent_registration_v1"
            / "ring_015_cycle_registration.png",
            "vggt_projection_noise_report": self.phase_root
            / "vggt_projection_noise_v1"
            / "report.json",
            "vggt_projection_noise_ring_004": self.phase_root
            / "vggt_projection_noise_v1"
            / "ring_004_projection_noise.png",
            "vggt_projection_noise_ring_020": self.phase_root
            / "vggt_projection_noise_v1"
            / "ring_020_projection_noise.png",
            "vggt_projection_noise_ring_007": self.phase_root
            / "vggt_projection_noise_v1"
            / "ring_007_projection_noise.png",
            "vggt_projection_noise_ring_015": self.phase_root
            / "vggt_projection_noise_v1"
            / "ring_015_projection_noise.png",
            "vggt_projection_noise_ring_023": self.phase_root
            / "vggt_projection_noise_v1"
            / "ring_023_projection_noise.png",
            "vggt_ring_001_reprojection": self.phase_root
            / "vggt_multiring_224_v1"
            / "ring_001"
            / "ring_001_cross_view_reprojection.png",
            "vggt_ring_007_reprojection": self.phase_root
            / "vggt_multiring_224_v1"
            / "ring_007"
            / "ring_007_cross_view_reprojection.png",
            "vggt_ring_023_reprojection": self.phase_root
            / "vggt_multiring_224_v1"
            / "ring_023"
            / "ring_023_cross_view_reprojection.png",
            "semantic_ring_001_holdout": self.phase_root
            / "semantic_camera_priors_v1"
            / "ring_001"
            / "ring_001_semantic_camera_holdout.png",
            "semantic_ring_007_holdout": self.phase_root
            / "semantic_camera_priors_v1"
            / "ring_007"
            / "ring_007_semantic_camera_holdout.png",
            "semantic_ring_023_holdout": self.phase_root
            / "semantic_camera_priors_v1"
            / "ring_023"
            / "ring_023_semantic_camera_holdout.png",
            "semantic_ring_001_reprojection": self.phase_root
            / "vggt_semantic_camera_refinement_v1"
            / "ring_001"
            / "ring_001_semantic_camera_reprojection.png",
            "ring01_depth_validation_report": self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "depth_validation_v1"
            / "report.json",
            "ring01_front_depth_photo": self.repo_root
            / self.phase1_depth_validation["front_photo_with_depth"],
            "ring01_front_depth_audit": self.repo_root
            / self.phase1_depth_validation["front_audit"],
            "ring01_all_views_depth_validation": self.repo_root
            / self.phase1_depth_validation["all_views_audit"],
            "ring01_pixel_feature_report": self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "pixel_features_v1"
            / "report.json",
            "ring01_pixel_feature_audit": self.repo_root
            / self.phase1_pixel_features["audit_sheet"],
            "phase1_ring01_review_notes": self.repo_root
            / "data"
            / "ring01_ground_truth_v1"
            / "review"
            / "ring01_review_notes.json",
            "phase2_paired_manifest": self.repo_root
            / "dataset"
            / "paired_cad_v1"
            / "manifest.json",
            "phase2_paired_report": self.repo_root
            / "dataset"
            / "paired_cad_v1"
            / "report.json",
            "phase2_image_only_report": self.repo_root
            / "data"
            / "ring01_image_only"
            / "reflection_correction_v1"
            / "report.json",
            "phase2_reflection_audit": self.repo_root
            / "data"
            / "ring01_image_only"
            / "reflection_correction_v1"
            / "ring01_angled_reflection_audit.png",
            "phase2_machine_review": self.repo_root
            / "data"
            / "ring01_image_only"
            / "machine_review_v1"
            / "report.json",
            "phase2_machine_review_notes": self.repo_root
            / "data"
            / "ring01_image_only"
            / "machine_review_v1"
            / "MACHINE_REVIEW.md",
            "metric_depth_consensus_report": self.phase_root
            / "image_only_phase2_metric_depth_consensus_v1"
            / "report.json",
            "metric_depth_ring01_preview": self.repo_root
            / self.metric_depth_consensus["review_sheets"]["ring01_reference"],
            "metric_depth_ring_001_preview": self.repo_root
            / self.metric_depth_consensus["review_sheets"]["ring_001"],
            "metric_depth_ring_004_preview": self.repo_root
            / self.metric_depth_consensus["review_sheets"]["ring_004"],
            "metric_depth_ring_007_preview": self.repo_root
            / self.metric_depth_consensus["review_sheets"]["ring_007"],
            "surface_normal_report": self.phase_root
            / "surface_normal_evidence_v1"
            / "report.json",
            "surface_normal_ring01_preview": self.repo_root
            / self.surface_normal_evidence["review_sheets"]["ring01_reference"],
            "surface_normal_ring_001_preview": self.repo_root
            / self.surface_normal_evidence["review_sheets"]["ring_001"],
            "surface_normal_ring_004_preview": self.repo_root
            / self.surface_normal_evidence["review_sheets"]["ring_004"],
            "surface_normal_ring_007_preview": self.repo_root
            / self.surface_normal_evidence["review_sheets"]["ring_007"],
            "external_ring_report": self.repo_root
            / "dataset"
            / "external_ring_v1"
            / "report.json",
            "external_ring_registry": self.repo_root
            / "dataset"
            / "external_sources"
            / "registry.json",
            "external_ornasia_preview": self.repo_root
            / "dataset"
            / "external_ring_v1"
            / "previews"
            / "ornasia_phase1_phase2_preview.jpg",
            "external_syr_preview": self.repo_root
            / "dataset"
            / "external_ring_v1"
            / "previews"
            / "segment_your_ring_phase1_phase2_preview.jpg",
            "upload_auditor_doc": self.repo_root
            / "docs"
            / "legacy_pipeline"
            / "UPLOAD_AUDITOR.md",
        }
        if name in reproducibility:
            return self._safe_path(reproducibility[name])
        try:
            filename = allowed[name]
        except KeyError as exc:
            raise AssetNotFoundError(f"Unknown global artifact: {name}") from exc
        return self._safe_path(self.phase_root / "comparisons" / filename)

    def export_object_report(self, object_id: str) -> dict[str, Any]:
        """Return a portable evidence report without leaking local absolute paths."""

        detail = self.object_detail(object_id)
        detail["schema_version"] = "lalitha_studio_evidence_v1"
        detail["claims"] = {
            "phase1_relative_depth_proposals_generated": True,
            "phase1_depth_accuracy_validated": False,
            "phase2_pseudo_label_self_consistency_measured": True,
            "phase2_human_ground_truth_accuracy_validated": False,
            "phase3_experimental_non_metric": True,
            "phase3_colmap_baseline_executed": bool(
                detail.get("geometry_evidence")
            ),
            "phase3_camera_pose_accuracy_validated": False,
            "phase3_vggt_executed": bool(
                self.phase3_geometry.get("vggt", {}).get("objects")
            ),
            "metric_reconstruction_validated": False,
            "manufacturing_accuracy_validated": False,
        }
        detail["benchmark_record"] = {
            "capture_tier": "A",
            "input_hashes_recorded": True,
            "code_commit_recorded_in_source_report": bool(
                self.phase0_environment.get("source", {}).get("commit")
            ),
            "model_checkpoint_revision_recorded": bool(
                self.phase1_depth.get("model", {}).get("revision")
            ),
            "model_checkpoint_hash_recorded": False,
            "license_manifest_recorded": True,
            "peak_memory_recorded": bool(
                self.phase1_depth.get("runtime", {}).get("peak_gpu_memory_bytes")
            ),
            "runtime_recorded": bool(
                self.phase1_depth.get("runtime", {}).get("total_seconds")
            ),
            "random_seed_recorded": False,
            "completeness": "partial",
            "decision": "research_only",
        }
        return detail


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))
