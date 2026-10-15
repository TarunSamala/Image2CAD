# Phase 4 - universal component and constraint graph

This implementation establishes the deterministic structural contract required
before exact CAD generation. It does not claim that Ring01 components are
reviewed or that local CAD regeneration is already valid.

## Implemented

- stable `component_id` and `track_id` fields;
- semantic class, confidence, provenance and review state;
- visible/amodal observation slots for every view;
- parent and support references;
- pose, dimensions and uncertainty fields;
- geometry-generator and free-form-surface links;
- manufacturing roles and dependency rules;
- directed stone-to-seat/cavity/prong dependencies;
- schema validation and dependency-cycle rejection;
- in-memory remove, resize and replace transactions;
- dry-run Flask API and Studio inspection controls.

Ring01 currently contains ten proposal nodes: assembly root, shank, setting,
stone, four prongs, seat and cavity. The Phase 1 component catalog supplies the
seven named jewellery components. Seat and cavity are explicit unknown
placeholders required by the structural contract. All ten remain pending review
and geometry generation is not authorized.

The four prongs now have stable machine-proposed IDs and separate masks in the
front, top and angled views. Every resolved proposal is disjoint and its union
exactly reproduces the retained aggregate mask. Side and back views collapse
physical prong pairs in projection, so those observations remain explicit
two-member overlap groups. No hidden boundary or reviewed identity is claimed.

The retained mask audit is
`data/phase4_prong_instances_v1/ring01_prong_instance_audit.png`. A second
VGGT audit samples the joint non-metric point maps inside every resolved mask
and evaluates all 24 identity permutations. It independently selects the same
ordering with strong top-view support (`0.816` permutation margin) and moderate
angled-view support (`0.147`). This is machine agreement, not reviewed truth.

The active graph checkpoint is versioned under
`data/phase4_component_graph_v3/`; v1 and v2 are preserved for backtracking.

## STL-1 component-candidate inventory

All 24 STL-1 objects and all 120 prepared views now have an enclosed
negative-space inventory under
`data/phase4_stl1_component_candidates_v1/`. The deterministic image-space
pass retains separate masks for structural openings, circular apertures and
decorative openwork. It found 266 disjoint view-local candidates.

The renders are monochrome and have no component labels. A circular region may
be an opening, bezel interior, missing insert or ornament, so this checkpoint
does not call any candidate a gemstone and does not infer seats, cavities or
cross-view identity. The front-view contact sheet is
`stl1_front_component_candidate_audit.png` and is available in Studio.

The separate `review_manifest.json` adds identified, append-only image-space
decisions without modifying proposal masks. Reviewers can classify a region as
an aperture, decorative openwork, structural opening, possible stone/setting,
or unknown. Even an approved possible-stone label keeps gemstone confirmation,
cross-view identity and CAD authorization false.

## Synthetic ten-stone benchmark

The exact synthetic graph contains ten stones, ten seats, ten cavities and four
prongs per stone. Every stone is exercised with three graph operations:

- remove;
- resize;
- replace.

All 30 cases pass the graph-state checks:

- component IDs remain stable;
- the changed-node set exactly matches the dependency closure;
- unrelated node state remains byte-stable under canonical hashing;
- the edited graph remains schema-valid.

This validates graph dependency locality only. It does not validate CadQuery or
OpenCascade regeneration, Boolean cavities, separate solids, collisions or
manufacturing rules. Therefore the Bible Phase 4 exit gate remains blocked.

## API

```text
GET  /api/phase4/graphs/ring01
GET  /api/phase4/graphs/synthetic-ten-stone
POST /api/phase4/graphs/synthetic-ten-stone/edit-preview
GET  /api/phase4/stl1-candidates
GET  /api/phase4/stl1-candidates/<object_id>
GET  /api/phase4/stl1-candidates/<object_id>/<view>/<kind>
POST /api/phase4/stl1-candidates/review
```

The edit endpoint is non-persistent and returns `geometry_generated: false`.
The request body is:

```json
{
  "component_id": "stone_007",
  "operation": "resize",
  "parameters": {"diameter": 2.8, "depth": 1.5}
}
```

## Reproduce

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase4_prong_instances

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.evaluate_phase4_component_tracks

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase4_stl1_component_candidates

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.phase4_candidate_review

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m pipeline.build_phase4_component_graph

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase4_prong_instances tests.test_phase4_component_tracks tests.test_phase4_stl1_component_candidates tests.test_phase4_candidate_review tests.test_phase4_component_graph
```

## Next gate

1. Review or correct Ring01 component identities and the three resolved prong
   views without forcing side/back overlap into false instances.
2. Review the STL-1 aperture/openwork inventory and add labelled real-image
   multi-stone evidence before producing stone/support instance masks.
3. Attach reviewed cross-view observations to stable track IDs.
4. Connect graph transactions to exact component generators.
5. Repeat the ten-stone benchmark on exported CAD and verify that unrelated
   solids and dimensions remain unchanged.
