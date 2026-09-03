# Phase 3.3.1 experiment record

## Accepted: projection-preserving cubic claws

- Retains the accepted Phase 3.3 shank, shoulders, gallery, stone, and camera fit.
- Gives every claw a stable ID: `prong_ne`, `prong_nw`, `prong_sw`, and `prong_se`.
- Converts the former quadratic post approximation into a smooth cubic centerline with a raised crest and inward tip.
- Builds each variable-radius claw as one smooth CadQuery loft rather than a chain of conical segments.
- Produces one connected valid metal B-rep, one separate valid gemstone B-rep, and a watertight derived print mesh.
- Exact validation: silhouette IoU `0.819726`, detail IoU `0.787388`, boundary F1 `0.795969`.

## Rejected: unconstrained instance-mask fitting

An optimizer weighted automatically extracted front/top prong instances at 18%. It raised its proxy objective, but enlarged and moved the claw heads outward. Exact mean silhouette IoU fell to `0.785103`, detail IoU to `0.733825`, and boundary F1 to `0.759235`. The exported STEP also contained five metal solids because the four claws were detached.

This direction was backtracked. Reflective machine-generated prong masks remain diagnostic only until human-reviewed instance masks are available.

## Rejected: segmented cubic tubes

The visually conservative cubic centerline scored well, but joining many tapered cone segments produced an invalid Boolean seam. Replacing that representation with a smooth multi-section loft retained the silhouette and restored a valid connected B-rep.

## Remaining limitation

The `0.90` target is not reached. Physical scale and hidden geometry remain uncalibrated, so this checkpoint is not manufacturing-accuracy validation.
