# Phase 1 — STL-1 Human Review

STL-1 is now the primary Phase 1 image-evidence review dataset. It contains 24
rings and five named views per ring, for 120 source images in total. The
machine extraction stage is complete and reproducible; the human acceptance
gate remains blocked until the retained proposals are inspected.

## What is ready

Every view has:

- the immutable source image;
- a normalized 768 × 768 image;
- a jewellery silhouette proposal;
- an OpenCV edge/detail proposal;
- a Depth Anything V2 relative-depth proposal;
- a transform-consistency uncertainty map;
- a 22-matrix pixel feature bundle;
- an exact source-pixel roundtrip check;
- three deterministic depth runs and horizontal/vertical diagnostics.

The automated contract covers all 24 objects and all 120 views. No source,
feature bundle or machine proposal is silently converted into human truth.

## Human exit gate

The dataset review manifest is:

`dataset/phase_runs/v1/phase1_human_review_v1/manifest.json`

For each of the 120 views, an identified reviewer must inspect seven evidence
categories:

1. source image quality;
2. normalization;
3. view semantics;
4. jewellery silhouette;
5. edges and fine-detail retention;
6. relative-depth plausibility;
7. depth uncertainty.

This produces 840 evidence decisions. Each of the 24 rings also requires one
cross-view component inventory covering visible stones, stone count, prongs,
sculptural relief and identity consistency. The gate passes only when all 840
evidence records and all 24 object inventories are approved and the automated
checks still pass.

## Review in Lalitha Studio

```bash
python -m pip install -r requirements-studio.txt
python -m studio.app
```

Open `http://127.0.0.1:8501`. The **What you can work on now** dashboard
shows live Ring01, STL-1 and ambiguous-region review counts. Use **Open next
pending review** or a ring in the 24-object queue to jump to the existing
review controls. The progress bar measures only identified human decisions;
it is not a CAD-accuracy or manufacturing-completion percentage.

Select a ring, then open **Phase 1 review**.
Enter the reviewer name, inspect each view/evidence pair and record a decision.
For feedback that is not yet a decision, enter a note and choose **Save comment
only**. The comment is attached to the selected evidence item or ring inventory,
shown when that item is revisited, and does not approve it or advance the gate.
Prior decision notes remain visible alongside these comments.
Use **Next pending item** to move through the selected ring.
After inspecting all seven categories for one view, **Approve all 7 inspected
items in this view** records the same identified approval on those seven items
in one atomic update. It does not approve any other view or ring.

If a silhouette is inaccurate, mark it **Needs correction**. A corrected binary
PNG can be uploaded from the same panel. It must be 768 × 768 and contain both
foreground and background. The original proposal remains unchanged; the saved
correction returns to `pending` and must be inspected and approved separately.

## What approval means

Approval means that the retained evidence is accepted in image space. It does
not establish:

- metric depth or millimetres;
- calibrated camera intrinsics/extrinsics;
- hidden or backside geometry;
- individual stone, prong, cavity or relief masks;
- production CAD, scan truth or manufacturing accuracy.

STL-1 is sufficient for completing this Phase 1 visual-review gate. A separate
paired photo-to-production-CAD dataset remains necessary for later metric,
structural and manufacturing validation.
