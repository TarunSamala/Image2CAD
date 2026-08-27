# Upstream datasets

The original general-purpose CAD reconstruction code supports the following datasets:

- DeepCAD test meshes
- Fusion360 test meshes
- Text2CAD train, validation and test data
- CAD-Recode train and validation data

The datasets were distributed through the upstream Hugging Face repositories and are separate from the Ring01 jewellery artifacts in this repository. Before using the upstream training/evaluation paths, install Git LFS and follow the dataset-specific licenses and preparation instructions from their source projects.

The expected upstream dataset layout is:

```text
data/
├── cad-recode-v1.5/
├── text2cad/
├── deepcad_test_mesh/
└── fusion360_test_mesh/
```

`data/cadrecode2mesh.py` is retained for converting CAD-Recode CadQuery programs to meshes.
