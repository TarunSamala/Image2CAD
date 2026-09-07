# Jewellery Phase Auditor

The auditor accepts one image or a complete five-view set and writes a self-contained report without changing the uploaded files.

## Browser interface

Install the app dependencies in the project environment, then start the local interface:

```bash
pip install -r requirements-app.txt
python app.py
```

Open `http://127.0.0.1:7860`. Upload one image in any slot, or provide front, top, isometric, left-side and right-side images. The page returns a summary, detailed audit images, the process explanation, the JSON report, and a ZIP containing every artifact.

Docker option:

```bash
docker build -f Dockerfile.auditor -t image2cad-auditor:local .
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:7860:7860 \
  -e IMAGE2CAD_SERVER_NAME=0.0.0.0 \
  -v "$PWD/runs:/workspace/runs" \
  image2cad-auditor:local python app.py
```

## Command line

Single image:

```bash
PYTHONPATH=pipeline python pipeline/audit_jewellery.py \
  --image /path/to/jewellery.jpg \
  --name sample
```

Five views:

```bash
PYTHONPATH=pipeline python pipeline/audit_jewellery.py \
  --front /path/to/front.png \
  --top /path/to/top.png \
  --iso /path/to/isometric.png \
  --lsv /path/to/left.png \
  --rsv /path/to/right.png \
  --name sample
```

Results are versioned under `runs/uploads/` by default. Use `--output-dir` for a specific empty directory.

## What each phase means

- Phase 1 normalizes the image and extracts a whole-object silhouette, edges, contours, holes, occupancy and symmetry.
- Phase 2 refines the silhouette and records bright-detail and shadow proposals without pretending that reflections are confirmed stones or prongs.
- Phase 2.2 compares the two machine masks. Its IoU is stage agreement, not accuracy against human ground truth.
- Phase 3 runs only with all five named views. It exports a watertight non-metric visual hull as STL and 3MF and compares its reprojections with the refined masks.
- Later exact-CAD phases are reported as unavailable until component masks, camera calibration, physical scale and topology-specific fitting are supplied.

`auto` prefers SAM 2.1 Tiny when its checkpoint and package are available, otherwise uses the trained Tiny Jewellery U-Net, then OpenCV as the final fallback. Select `opencv` for a lightweight run or `sam` when processing difficult real photographs.
