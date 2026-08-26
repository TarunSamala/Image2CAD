# Phase 3.3 experiment log

## Accepted direction: angled-view perspective

- Orthographic exported-CAD mean IoU: 0.679256
- Perspective-angled exported-CAD mean IoU: 0.704854
- Perspective-angled IoU: 0.593210
- Decision: retain finite-distance perspective for the angled reference only.

## Rejected: one shared camera distance for all views

- Proxy balanced objective: 0.554328, below the accepted perspective-angled candidate at 0.561208.
- Front silhouette IoU fell to 0.532439.
- The independent product renders do not support a shared-intrinsics assumption.
- Decision: backtrack. Do not distort geometry to compensate for independently framed source views.
