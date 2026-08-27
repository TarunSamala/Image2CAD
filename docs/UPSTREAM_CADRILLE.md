# Upstream CAD reconstruction research

This repository began from the implementation of **cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning**. The original code paths remain at the repository root:

- `cadrille.py`
- `dataset.py`
- `train.py`
- `test.py`
- `evaluate.py`
- `Dockerfile`

The upstream project supports point-cloud, image and text modalities and experiments on DeepCAD, Fusion360, Text2CAD and CAD-Recode datasets. The jewellery-specific work in `pipeline/` builds on this research environment but has a separate phased workflow and validation policy.

## Original usage

Training:

```bash
python train.py --mode pc_img --use-text
```

Inference:

```bash
python test.py --split deepcad_test_mesh --mode pc
```

Evaluation:

```bash
python evaluate.py
```

## Paper

Maksim Kolodiazhnyi et al., *cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning*, arXiv:2505.22914 (2025).

```bibtex
@article{kolodiazhnyi2025cadrille,
  title={cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning},
  author={Maksim Kolodiazhnyi and Denis Tarasov and Dmitrii Zhemchuzhnikov and Alexander Nikulin and Ilya Zisman and Anna Vorontsova and Anton Konushin and Vladislav Kurenkov and Danila Rukhovich},
  journal={arXiv preprint arXiv:2505.22914},
  year={2025}
}
```
