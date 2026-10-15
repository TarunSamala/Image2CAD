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

The four prongs each have a stable proposed ID, but currently point to the same
aggregate prong mask in every view. The graph marks this explicitly and does not
pretend that individual prong segmentation exists.

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
  python -m pipeline.build_phase4_component_graph

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  image2cad-validation:phase0 \
  python -m unittest tests.test_phase4_component_graph
```

## Next gate

1. Review or correct Ring01 component identities and masks.
2. Produce individual stone/prong/support instance masks.
3. Attach reviewed cross-view observations to stable track IDs.
4. Connect graph transactions to exact component generators.
5. Repeat the ten-stone benchmark on exported CAD and verify that unrelated
   solids and dimensions remain unchanged.
