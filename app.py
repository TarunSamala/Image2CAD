"""Local browser interface for one-image or five-view jewellery phase audits."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

from audit_jewellery import FIVE_VIEWS, _safe_name, run_audit  # noqa: E402


def process(name: str, segmenter: str, front: str | None, top: str | None, iso: str | None, left: str | None, right: str | None):
    supplied = {key: Path(value) for key, value in zip(FIVE_VIEWS, (front, top, iso, left, right)) if value}
    if len(supplied) == 1:
        only = next(iter(supplied.values()))
        supplied = {"single": only}
    elif len(supplied) != 5:
        raise gr.Error("Upload exactly one image, or upload all five named views.")
    output = ROOT / "runs" / "uploads" / f'{datetime.now():%Y%m%d-%H%M%S-%f}-{_safe_name(name)}'
    try:
        report = run_audit(supplied, output, segmenter_name=segmenter)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    previews = [(path, Path(path).stem.replace("_", " ").title()) for path in report["artifacts"]["view_audits"]]
    if report["phase3"].get("exports"):
        previews.append((report["phase3"]["exports"]["comparison"], "Phase 3 comparison"))
    return report["artifacts"]["summary"], previews, Path(report["artifacts"]["process_explanation"]).read_text(encoding="utf-8"), json.dumps(report, indent=2), archive


with gr.Blocks(title="Jewellery Phase Auditor") as demo:
    gr.Markdown("# Jewellery Phase Auditor\nUpload one image for Phase 1-2.2, or all five views to include an experimental Phase 3 visual hull.")
    with gr.Row():
        name = gr.Textbox(label="Audit name", value="jewellery")
        segmenter = gr.Dropdown(("auto", "sam", "unet", "opencv"), value="auto", label="Segmentation backend")
    with gr.Row():
        front = gr.Image(type="filepath", label="Front / single image")
        top = gr.Image(type="filepath", label="Top")
        iso = gr.Image(type="filepath", label="Isometric")
        left = gr.Image(type="filepath", label="Left side")
        right = gr.Image(type="filepath", label="Right side")
    run = gr.Button("Run phase audit", variant="primary")
    summary = gr.Image(label="All phases summary")
    gallery = gr.Gallery(label="Detailed audits", columns=1)
    explanation = gr.Markdown()
    raw_report = gr.Code(label="Machine-readable report", language="json")
    download = gr.File(label="Download complete audit ZIP")
    run.click(process, (name, segmenter, front, top, iso, left, right), (summary, gallery, explanation, raw_report, download))


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("IMAGE2CAD_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("IMAGE2CAD_SERVER_PORT", "7860")),
        show_error=True,
    )
