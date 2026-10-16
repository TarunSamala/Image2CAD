"""Regression checks for retained Phase 4 exact-CAD locality evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


class Phase4ExactCadLocalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("data/phase4_exact_cad_locality_v1")
        cls.report = json.loads(
            (cls.root / "report.json").read_text(encoding="utf-8")
        )

    def test_all_operations_pass_for_every_stone(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["case_count"], 30)
        self.assertEqual(summary["passed_case_count"], 30)
        self.assertEqual(summary["stone_count"], 10)
        self.assertEqual(summary["operations"], ["remove", "replace", "resize"])
        self.assertEqual(
            {case["edited_component_id"] for case in self.report["cases"]},
            {f"stone_{index:03d}" for index in range(1, 11)},
        )

    def test_exact_component_locality_is_preserved(self) -> None:
        for case in self.report["cases"]:
            with self.subTest(
                component=case["edited_component_id"], operation=case["operation"]
            ):
                self.assertTrue(case["passed"])
                self.assertEqual(
                    case["changed_exact_component_ids"],
                    case["expected_affected_component_ids"],
                )
                self.assertTrue(
                    case["checks"]["unrelated_brep_hashes_unchanged"]
                )
                self.assertTrue(case["checks"]["derived_metal_valid"])
                self.assertTrue(case["checks"]["derived_metal_single_solid"])

    def test_baseline_has_independent_exact_components(self) -> None:
        signatures = self.report["baseline"]["component_signatures"]
        self.assertEqual(len(signatures), 71)
        self.assertTrue(signatures["metal_body_001"]["valid"])
        self.assertEqual(
            len([key for key in signatures if key.startswith("stone_")]), 10
        )
        self.assertEqual(
            len([key for key in signatures if key.startswith("prong_")]), 40
        )
        for signature in signatures.values():
            if signature["active"]:
                self.assertRegex(signature["brep_sha256"], r"^[0-9a-f]{64}$")

    def test_exports_and_claim_boundaries_are_retained(self) -> None:
        artifacts = self.report["artifacts"]
        retained = [artifacts["audit"], *artifacts["baseline"].values()]
        for operation in ("remove", "replace", "resize"):
            retained.extend(artifacts["stone_007"][operation].values())
        for relative_path in retained:
            path = Path(relative_path)
            self.assertTrue(path.is_file(), relative_path)
            self.assertGreater(path.stat().st_size, 0, relative_path)
        self.assertTrue(
            self.report["claims"]["synthetic_exact_cad_locality_validated"]
        )
        self.assertFalse(
            self.report["claims"]["ring01_image_derived_cad_validated"]
        )
        self.assertFalse(
            self.report["claims"]["manufacturing_accuracy_validated"]
        )


if __name__ == "__main__":
    unittest.main()
