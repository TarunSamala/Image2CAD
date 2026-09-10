"""Tests for the category-independent Phase 2.3 evidence graph."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from extract_phase2_3 import _foreground_quality, extract_phase2_3


class Phase23EvidenceTest(unittest.TestCase):
    def _input(self, root: Path, view: str, shift: int = 0) -> dict[str, Path]:
        image = np.full((192, 192, 3), 244, np.uint8)
        mask = np.zeros((192, 192), np.uint8)
        cv2.ellipse(mask, (96 + shift, 104), (57, 68), 0, 0, 360, 255, 15)
        cv2.circle(mask, (96 + shift, 48), 25, 255, -1)
        image[mask > 0] = (125, 135, 145)
        for x, y in ((84 + shift, 42), (108 + shift, 42), (84 + shift, 58), (108 + shift, 58)):
            cv2.circle(image, (x, y), 6, (235, 235, 235), -1)
            cv2.circle(image, (x, y), 7, (55, 55, 55), 1)
        image_path, mask_path = root / f"{view}.png", root / f"{view}_mask.png"
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)
        return {"image": image_path, "foreground": mask_path}

    def test_outputs_reviewable_evidence_without_semantic_overclaim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = extract_phase2_3({"single": self._input(root, "single")}, root / "phase2_3")
            self.assertGreater(len(report["views"]["single"]["observations"]), 0)
            self.assertFalse(report["validation"]["accuracy_claim_allowed"])
            self.assertFalse(report["validation"]["human_review_complete"])
            self.assertTrue(all(item["kind"] == "generic_detail" for item in report["views"]["single"]["observations"]))
            review = json.loads(Path(report["review_manifest"]).read_text(encoding="utf-8"))
            self.assertIn("stone", review["allowed_semantics"])
            self.assertTrue(all(item["decision"] == "pending" for item in review["decisions"].values()))

    def test_cross_view_tracks_have_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = ("front", "top", "iso", "lsv", "rsv")
            inputs = {view: self._input(root, view, index - 2) for index, view in enumerate(views)}
            report = extract_phase2_3(inputs, root / "phase2_3")
            self.assertGreater(len(report["cross_view_tracks"]), 0)
            observations = [item for view in report["views"].values() for item in view["observations"]]
            self.assertTrue(all(item["track_id"] for item in observations))
            self.assertTrue(all(track["state"] in {"machine_hypothesis", "single_view_hypothesis"} for track in report["cross_view_tracks"]))

    def test_nearly_rectangular_foreground_requires_review(self) -> None:
        mask = np.zeros((192, 192), np.uint8)
        mask[16:176, 14:178] = 255
        quality = _foreground_quality(mask)
        self.assertFalse(quality["passed"])
        self.assertIn("foreground_is_nearly_solid_bounding_rectangle", quality["reasons"])

    def test_refuses_to_overwrite_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "phase2_3"
            output.mkdir()
            (output / "keep.txt").write_text("checkpoint", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                extract_phase2_3({"single": self._input(root, "single")}, output)


if __name__ == "__main__":
    unittest.main()
