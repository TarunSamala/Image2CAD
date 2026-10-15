"""Regression gates for the Bible Phase 4 component graph foundation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from pipeline.component_constraint_graph import (
    ComponentGraphError,
    apply_local_edit,
    dependency_closure,
    node_hashes,
    validate_graph,
)


class Phase4ComponentGraphTest(unittest.TestCase):
    root = Path("data/phase4_component_graph_v1")

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (cls.root / "report.json").read_text(encoding="utf-8")
        )
        cls.ring01 = json.loads(
            (cls.root / "ring01_component_graph.json").read_text(encoding="utf-8")
        )
        cls.synthetic = json.loads(
            (cls.root / "synthetic_ten_stone_graph.json").read_text(
                encoding="utf-8"
            )
        )

    def test_graph_contracts_are_valid_and_ring01_stays_review_gated(self) -> None:
        self.assertEqual(validate_graph(self.ring01), [])
        self.assertEqual(validate_graph(self.synthetic), [])
        self.assertEqual(len(self.ring01["nodes"]), 10)
        self.assertEqual(len(self.ring01["edges"]), 15)
        self.assertFalse(self.ring01["gate"]["reviewed_component_graph"])
        self.assertFalse(self.ring01["gate"]["individual_prong_masks_available"])
        self.assertFalse(self.ring01["gate"]["seat_and_cavity_observed"])
        self.assertFalse(self.ring01["gate"]["geometry_generation_authorized"])
        self.assertTrue(
            all(node["review_state"] == "pending" for node in self.ring01["nodes"])
        )

    def test_ring01_prongs_use_aggregate_masks_without_false_instance_claim(self) -> None:
        prongs = [
            node for node in self.ring01["nodes"]
            if node["semantic_class"] == "prong"
        ]
        self.assertEqual(len(prongs), 4)
        for prong in prongs:
            self.assertEqual(set(prong["observations"]), {
                "front", "side", "top", "angled", "back"
            })
            self.assertTrue(
                all(
                    observation["aggregate_instance_mask"]
                    for observation in prong["observations"].values()
                )
            )
            self.assertEqual(prong["provenance_state"], "inferred")

    def test_stone_seven_dependency_closure_is_local_and_complete(self) -> None:
        self.assertEqual(
            dependency_closure(self.synthetic, "stone_007"),
            [
                "cavity_007",
                "prong_025",
                "prong_026",
                "prong_027",
                "prong_028",
                "seat_007",
                "stone_007",
            ],
        )
        before = node_hashes(self.synthetic)
        edited, audit = apply_local_edit(
            deepcopy(self.synthetic),
            "stone_007",
            "resize",
            {"diameter": 2.8, "depth": 1.5},
        )
        after = node_hashes(edited)
        self.assertTrue(audit["stable_component_ids"])
        self.assertTrue(audit["affected_set_exact"])
        self.assertTrue(audit["unrelated_geometry_state_unchanged"])
        self.assertTrue(audit["graph_valid_after_edit"])
        self.assertEqual(before["stone_006"], after["stone_006"])
        self.assertNotEqual(before["seat_007"], after["seat_007"])

    def test_all_ten_stones_pass_remove_resize_replace_graph_edits(self) -> None:
        benchmark = self.report["synthetic_ten_stone_edit_benchmark"]
        self.assertEqual(benchmark["case_count"], 30)
        self.assertEqual(benchmark["passed_case_count"], 30)
        self.assertEqual(benchmark["pass_fraction"], 1.0)
        self.assertTrue(benchmark["all_stone_ids_exercised"])
        self.assertEqual(benchmark["operations"], ["remove", "replace", "resize"])
        self.assertTrue(all(case["passed"] for case in benchmark["cases"]))

    def test_report_does_not_overstate_phase4_or_cad_readiness(self) -> None:
        checks = self.report["checks"]
        self.assertTrue(checks["synthetic_ten_stone_edit_test_passed"])
        self.assertTrue(checks["no_unrelated_graph_state_changes"])
        self.assertTrue(checks["stable_ids_across_all_edits"])
        self.assertFalse(checks["reviewed_ring01_graph_available"])
        self.assertFalse(checks["phase4_exit_gate_passed"])
        self.assertFalse(any(self.report["claims"].values()))
        self.assertIn("graph-state", self.report["limitations"][0])

    def test_unknown_component_and_invalid_operation_fail_closed(self) -> None:
        with self.assertRaises(ComponentGraphError):
            apply_local_edit(self.synthetic, "stone_999", "resize", {})
        with self.assertRaises(ComponentGraphError):
            apply_local_edit(self.synthetic, "stone_001", "warp", {})


if __name__ == "__main__":
    unittest.main()
