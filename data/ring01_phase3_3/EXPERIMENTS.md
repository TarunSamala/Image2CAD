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

## Rejected: downweight semantic components

- The candidate increased emphasis on silhouette and boundary because the component masks are not human ground truth.
- It slightly improved front/top proxy scores but reduced side, angled and back agreement.
- Decision: backtrack. Retain confidence-weighted semantic components as structural regularization.

## Accepted: reviewed silhouette physics correction

- The raw masks incorrectly classified elongated specular highlights in the front/top opaque band as physical holes.
- Only those reviewed elongated regions were filled; ring, gallery and head openings remain open.
- Both reviewed-mask and untouched raw-machine-mask IoU are retained in validation.
- This remains reviewed machine supervision, not human ground truth.

## Accepted: resumable local refinement

- The optimizer resumed from the accepted `0.645038` checkpoint instead of restarting from defaults.
- Balanced proxy objective increased to `0.662347` with all five proxy silhouettes improving.
- Independently rendered exported-CAD metrics reached mean IoU `0.818114`, detail IoU `0.784277`, and external-boundary F1 `0.795217`.
- Exact STEP solids and the derived print mesh pass topology and watertightness checks.
- Decision: retain as the current Phase 3.3 checkpoint. Do not leave Phase 3 until human masks and metric scale are available.
