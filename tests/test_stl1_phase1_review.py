import json
import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.stl1_phase1_review import (
    EVIDENCE_TYPES,
    MANIFEST_RELATIVE_PATH,
    VIEWS,
    apply_evidence_review,
    apply_object_review,
    apply_view_review,
    add_review_comment,
    load_manifest,
    save_corrected_silhouette,
)


ROOT = Path(__file__).resolve().parents[1]


class STL1Phase1ReviewContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.manifest = load_manifest(ROOT)

    def test_all_twenty_four_rings_and_120_views_are_reviewable(self) -> None:
        self.assertEqual(len(self.manifest["objects"]), 24)
        self.assertEqual(self.manifest["gate"]["view_count"], 120)
        self.assertEqual(self.manifest["gate"]["evidence_review_count"], 840)
        for record in self.manifest["objects"].values():
            self.assertEqual(set(record["views"]), set(VIEWS))
            for view in VIEWS:
                self.assertEqual(
                    set(record["views"][view]["evidence"]),
                    set(EVIDENCE_TYPES),
                )

    def test_every_review_item_points_to_retained_evidence(self) -> None:
        unique_paths = {}
        for record in self.manifest["objects"].values():
            for view_record in record["views"].values():
                self.assertTrue(all(view_record["automated_checks"].values()))
                for review in view_record["evidence"].values():
                    path = ROOT / review["proposal_path"]
                    self.assertTrue(path.is_file(), path)
                    self.assertRegex(review["proposal_sha256"], r"^[0-9a-f]{64}$")
                    unique_paths[str(path)] = path
        self.assertGreaterEqual(len(unique_paths), 600)

    def test_gate_is_honestly_blocked_until_identified_human_review(self) -> None:
        gate = self.manifest["gate"]
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["approved_evidence_count"], 0)
        self.assertEqual(gate["pending_evidence_count"], 840)
        self.assertEqual(gate["pending_object_review_count"], 24)
        self.assertFalse(gate["metric_scale_available"])
        self.assertFalse(gate["manufacturing_accuracy_validated"])
        self.assertEqual(gate["automated_check_failures"], [])


class STL1Phase1ReviewMutationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        target = self.root / MANIFEST_RELATIVE_PATH
        target.parent.mkdir(parents=True)
        shutil.copy2(ROOT / MANIFEST_RELATIVE_PATH, target)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_review_decisions_preserve_history_and_require_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reviewer is required"):
            apply_evidence_review(
                self.root, "ring_001", "front", "silhouette", "approved", ""
            )
        manifest = apply_evidence_review(
            self.root,
            "ring_001",
            "front",
            "silhouette",
            "needs_correction",
            "Dataset Reviewer",
            "Outer edge needs cleanup",
        )
        review = manifest["objects"]["ring_001"]["views"]["front"]["evidence"][
            "silhouette"
        ]
        self.assertEqual(review["decision"], "needs_correction")
        self.assertEqual(review["reviewer"], "Dataset Reviewer")
        self.assertEqual(len(review["history"]), 1)

    def test_comment_is_scoped_and_does_not_change_review_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "Comment is required"):
            add_review_comment(self.root, "ring_001", "evidence", "Reviewer", "", "front", "silhouette")
        before = load_manifest(self.root)[1]["gate"]["pending_evidence_count"]
        updated = add_review_comment(
            self.root, "ring_001", "evidence", "Reviewer", "Inner reflection is not an edge",
            "front", "silhouette",
        )
        selected = updated["objects"]["ring_001"]["views"]["front"]["evidence"]["silhouette"]
        self.assertEqual(selected["decision"], "pending")
        self.assertEqual(selected["comments"][0]["text"], "Inner reflection is not an edge")
        self.assertEqual(updated["gate"]["pending_evidence_count"], before)
        inventory = add_review_comment(self.root, "ring_001", "object", "Reviewer", "Count stones carefully")
        self.assertEqual(inventory["objects"]["ring_001"]["object_review"]["comments"][0]["text"], "Count stones carefully")

    def test_object_inventory_is_addressable_per_ring(self) -> None:
        manifest = apply_object_review(
            self.root,
            "ring_001",
            "approved",
            "Dataset Reviewer",
            {
                "has_stones": "no",
                "estimated_visible_stone_count": 0,
                "has_prongs": "no",
                "has_sculptural_relief": "yes",
                "cross_view_identity_consistent": "yes",
            },
            "Five views inspected",
        )
        review = manifest["objects"]["ring_001"]["object_review"]
        self.assertEqual(review["decision"], "approved")
        self.assertEqual(review["component_inventory"]["has_sculptural_relief"], "yes")
        self.assertEqual(len(review["history"]), 1)

    def test_view_approval_updates_exactly_seven_evidence_records(self) -> None:
        manifest = apply_view_review(
            self.root,
            "ring_001",
            "front",
            "approved",
            "Dataset Reviewer",
            "All seven front-view layers inspected",
        )
        front = manifest["objects"]["ring_001"]["views"]["front"]["evidence"]
        top = manifest["objects"]["ring_001"]["views"]["top"]["evidence"]
        self.assertTrue(all(item["decision"] == "approved" for item in front.values()))
        self.assertTrue(all(len(item["history"]) == 1 for item in front.values()))
        self.assertTrue(all(item["decision"] == "pending" for item in top.values()))
        self.assertEqual(manifest["gate"]["approved_evidence_count"], 7)

    def test_corrected_mask_is_separate_binary_and_requires_reapproval(self) -> None:
        path, manifest = load_manifest(self.root)
        review = manifest["objects"]["ring_001"]["views"]["front"]["evidence"][
            "silhouette"
        ]
        proposal = self.root / "proposal.png"
        image = Image.new("L", (768, 768), 0)
        ImageDraw.Draw(image).ellipse((120, 100, 650, 700), fill=255)
        image.save(proposal)
        review["proposal_path"] = "proposal.png"
        review["active_path"] = "proposal.png"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        corrected = Image.new("L", (768, 768), 0)
        ImageDraw.Draw(corrected).ellipse((130, 110, 640, 690), fill=230)
        invalid_payload = BytesIO()
        corrected.save(invalid_payload, format="JPEG")
        with self.assertRaisesRegex(ValueError, "PNG"):
            save_corrected_silhouette(
                self.root,
                "ring_001",
                "front",
                invalid_payload.getvalue(),
                "Dataset Reviewer",
            )
        payload = BytesIO()
        corrected.save(payload, format="PNG")
        updated = save_corrected_silhouette(
            self.root,
            "ring_001",
            "front",
            payload.getvalue(),
            "Dataset Reviewer",
        )
        updated_review = updated["objects"]["ring_001"]["views"]["front"][
            "evidence"
        ]["silhouette"]
        self.assertEqual(updated_review["decision"], "pending")
        self.assertNotEqual(
            updated_review["proposal_path"], updated_review["corrected_path"]
        )
        output = self.root / updated_review["corrected_path"]
        self.assertTrue(output.is_file())
        with Image.open(output) as result:
            self.assertEqual(set(result.getdata()), {0, 255})


if __name__ == "__main__":
    unittest.main()
