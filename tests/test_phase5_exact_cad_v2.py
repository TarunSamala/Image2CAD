"""Regression checks for the retained Bible Phase 5 CAD generator fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


class Phase5ExactCadV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path("data/phase5_exact_cad_v2_v1")
        cls.report = json.loads(
            (cls.root / "report.json").read_text(encoding="utf-8")
        )

    def test_all_four_structural_families_pass(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["family_count"], 4)
        self.assertEqual(summary["passed_family_count"], 4)
        self.assertEqual(
            set(summary["families"]),
            {"round_solitaire", "oval_solitaire", "three_stone", "halo"},
        )
        self.assertTrue(self.report["checks"]["all_family_fixtures_passed"])

    def test_exact_topology_and_roundtrip_checks_pass(self) -> None:
        for family in self.report["families"]:
            with self.subTest(family=family["family"]):
                self.assertTrue(family["passed"])
                self.assertTrue(all(family["checks"].values()))
                self.assertTrue(family["metal"]["valid"])
                self.assertEqual(family["metal"]["solid_count"], 1)
                self.assertEqual(
                    family["assembly"]["solid_count"],
                    family["topology"]["stone_count"] + 1,
                )

    def test_multi_stone_components_remain_individually_indexed(self) -> None:
        indexed = {item["family"]: item for item in self.report["families"]}
        self.assertEqual(indexed["three_stone"]["topology"]["stone_count"], 3)
        self.assertEqual(indexed["halo"]["topology"]["stone_count"], 9)
        for family_name in ("three_stone", "halo"):
            family = indexed[family_name]
            ids = family["component_ids"]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(
                len([item for item in ids if item.startswith("stone_")]),
                family["topology"]["stone_count"],
            )
            self.assertEqual(
                len([item for item in ids if item.startswith("cavity_")]),
                family["topology"]["cavity_count"],
            )

    def test_artifacts_and_claim_boundaries_are_retained(self) -> None:
        paths = [self.report["artifacts"]["audit"]]
        for family in self.report["families"]:
            paths.extend(family["artifacts"].values())
        for relative_path in paths:
            path = Path(relative_path)
            self.assertTrue(path.is_file(), relative_path)
            self.assertGreater(path.stat().st_size, 0, relative_path)
        claims = self.report["claims"]
        self.assertTrue(claims["designed_exact_cad_generators_validated"])
        self.assertFalse(claims["image_derived_exact_cad_validated"])
        self.assertFalse(claims["unseen_image_family_generalization_validated"])
        self.assertFalse(claims["manufacturing_accuracy_validated"])


if __name__ == "__main__":
    unittest.main()
