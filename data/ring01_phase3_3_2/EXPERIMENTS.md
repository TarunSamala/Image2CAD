# Phase 3.3.2 correction record

## Diagnosis

The STEP model was not missing a fourth prong. It contained four named claw solids and the gem-face view showed all four. The original clean-preview camera viewed the ring along a 45-degree prong axis, causing the near claw to project through the gemstone centre and visually resemble a central post.

## Rejected: rotating the prong geometry

Rotating the four claws solely to improve one generic preview would move them away from the diagonal locations observed in the front and top reference images. That would trade correct geometry for a presentation-specific result.

## Accepted: reference-fitted inspection camera

- Reuses the fitted Ring01 angled-camera orientation.
- Places the camera between prong axes so no claw projects onto the centre axis.
- Verifies four distinct projected tip positions in isometric, opposite, and gem-face views.
- Preserves the Phase 3.3.1 STEP byte-for-byte and therefore preserves all exact image metrics and topology checks.

This phase corrects inspection and validation. It does not claim additional geometric accuracy or manufacturing validation.
