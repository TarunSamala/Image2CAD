# Repository structure

The repository keeps source code, inputs, generated artifacts, and validation tests separate.

```text
Image2CAD/
├── pipeline/                 Pipeline stages and reconstruction tools
├── tests/                    Regression and artifact-validation tests
├── docs/                     Project and repository documentation
├── dataset/
│   ├── STL-1/                Original 24-ring, five-view source dataset
│   ├── prepared_v1/          Normalized images, masks, edges, and split manifests
│   ├── Sample_test/          Independent sample-ring validation input and results
│   └── phase_runs/v1/        Versioned training and dataset-wide phase outputs
│       ├── checkpoints/      Trained model weights
│       ├── comparisons/      Dataset comparison sheets
│       ├── phase_audits/     Per-ring phase audit sheets
│       ├── test_predictions/ Held-out predictions
│       └── phase3/
│           └── visual_hull/  Non-metric STL, 3MF, previews, and reports
└── data/                     Ring01 development and historical phase artifacts
```

## Conventions

- Original inputs stay unchanged.
- Generated dataset experiments are grouped by run version and phase.
- Phase outputs include their reports and previews beside the corresponding models.
- `hf_cache/`, Python bytecode, logs, and downloaded model files are local runtime data and remain ignored by Git.
- Phase 3 visual hulls are non-metric validation artifacts, not manufacturing-ready CAD.
