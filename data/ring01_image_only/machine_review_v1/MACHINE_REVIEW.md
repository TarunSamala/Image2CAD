# Image-only Phase 2 machine-review checkpoint

Status: **RESEARCH CONTINUATION ALLOWED WITH BLOCKED HUMAN GATE**

This checkpoint records deterministic validation and AI-assisted visual inspection. It is not an identified human review and does not close Bible Phase 1 or canonical Bible Phase 2.

## Phase 1 gate carried forward

- Ring01 pending label reviews: `35`
- Ring01 pending component identities: `7`
- Ring01 pending camera records: `5`
- STL-1 pending evidence decisions: `840`
- STL-1 pending object inventories: `24`

## Ring01 machine validation

- Five views and `35` proposal masks checked.
- Source hashes valid: `True`
- Proposal hashes valid: `True`
- Binary, nonempty, source-sized masks: `True`

- All five whole-jewellery silhouettes preserve the visible ring body, head and principal negative spaces.
- Front and top proposals preserve the visible four-prong arrangement, but gemstone facets and prong boundaries remain reflection-sensitive.
- Side and back proposals preserve the shank opening and gallery silhouette; thin gallery and setting boundaries remain uncertain at source resolution.
- The angled legacy mask contains reflection-induced inner-shank gaps; the separate correction improves continuity without changing the exterior silhouette.
- Relative-depth ordering is visually plausible at coarse scale but uncertainty is high around gemstone facets, prongs and polished metal, so it is non-metric evidence only.

## STL-1 machine validation

- `24` objects, `120` views and `840` evidence records checked.
- Automated checks pass: `True`
- Evidence paths and hashes valid: `True`

- The 24-object, five-view overview is complete and view assignments are visually consistent with the source filenames.
- Normalization retains the full visible objects across the representative train, validation and test examples.
- Pseudo-silhouettes preserve outer jewellery contours but do not provide individual stones, prongs, cavities or sculptural-relief instances.
- OpenCV edge evidence retains substantial relief detail, including deity faces and ornament boundaries, but edge presence is not semantic reconstruction.
- Held-out overlap scores measure reproduction of machine pseudo-masks and are not human-ground-truth accuracy.

## Decision

Image-only Phase 2 research may continue using these artifacts as machine proposals with uncertainty. Human ground truth, metric depth, canonical paired-CAD Phase 2, exact hidden geometry and manufacturing validation remain blocked.
