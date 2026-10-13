# Cross-view backend readiness

This checkpoint records whether the candidate learned multi-view backends can
be used in the public, team-reviewed Studio. It performs no inference and does
not download or redistribute model checkpoints.

## Result

- **VGGT** is the only current public-product candidate. Its code licence is
  commercial-use-friendly, but only the gated `VGGT-1B-Commercial` checkpoint
  is eligible. The checkpoint must be requested and accepted by the operator.
- **VGGT-Omega** is under the FAIR Noncommercial Research License. Its official
  single-frame benchmark reports `6.02 GB` peak GPU memory, above this laptop's
  approximately `4 GB` GPU.
- **DUSt3R** and **MASt3R** are licensed under CC BY-NC-SA 4.0 and remain
  research-only candidates rather than public-product dependencies.

The retained machine-readable report is:

```text
dataset/phase_runs/v1/cross_view_backend_readiness_v1/report.json
```

## Low-VRAM research result

The conservative 8 GiB policy was tested rather than treated as an absolute
hardware limit. The original noncommercial `VGGT-1B` checkpoint was content
verified and run on Ring01 under an explicitly research-only profile:

- five images at `224 x 224`;
- FP16 camera, depth and world-point heads;
- depth and point heads processed one frame at a time;
- `2,724,175,360` peak allocated CUDA bytes (about `2.54 GiB`);
- `27.4731` seconds for the retained provenance-complete dense run;
- five finite camera matrices, depth maps and world-point maps produced.

The depth ordering is visually useful, but the fused point cloud contains
duplicated shank/shoulder surfaces and floating head-region outliers. It is
therefore accepted as camera/depth research evidence and rejected as CAD or
cross-view reconstruction geometry. The retained evidence is under
`dataset/phase_runs/v1/ring01_vggt_dense_224_v1/`.

## Decision

No learned cross-view backend is locally ready for the public product. A bounded
noncommercial VGGT research run is locally feasible on the 4 GiB GPU, but its
current fusion fails visual review. Product integration still requires the
operator to obtain and accept the commercial checkpoint. Higher-resolution
experiments should remain explicit and memory-bounded.
The existing COLMAP failure remains useful baseline evidence, but it is not a
substitute for learned cross-view geometry.

This is an engineering licence audit, not legal advice.

## Rebuild

```bash
python -m pipeline.run_cross_view_readiness --code-commit "$(git rev-parse HEAD)"
```

After the commercial checkpoint has been obtained and its terms accepted:

```bash
python -m pipeline.run_cross_view_readiness \
  --vggt-commercial-checkpoint /absolute/path/to/checkpoint \
  --vggt-license-accepted
```

Rebuild the retained research experiment only with the noncommercial checkpoint:

```bash
python -m pipeline.run_ring01_vggt_experiment \
  --checkpoint models/vggt/VGGT-1B/model.pt \
  --image-size 224 --full
python -m pipeline.validate_ring01_vggt_experiment
```
