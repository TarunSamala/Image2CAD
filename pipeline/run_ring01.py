"""Run a single Ring 01 multi-view inference through Cadrille."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from cadrille import Cadrille, collate


VIEW_NAMES = ("front", "side", "top", "angled", "back")


def run(
    input_dir: Path,
    output: Path,
    checkpoint: str,
    max_new_tokens: int,
    requested_device: str,
) -> None:
    views = [Image.open(input_dir / f"ring01_{name}.png").convert("RGB") for name in VIEW_NAMES]
    batch = [{
        "video": views,
        "description": (
            "Generate CadQuery code for Ring 01. It is a solitaire ring with one "
            "round-brilliant gemstone and four prongs."
        ),
        "file_name": "ring01",
    }]

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
        padding_side="left",
    )
    device = requested_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Cadrille.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
    )
    model.eval()

    inputs = collate(batch, processor=processor, n_points=256, eval=True)
    model_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(model.device)
        elif key not in {"file_name"}:
            model_inputs[key] = value

    with torch.inference_mode():
        generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    generated_trimmed = generated[0][len(inputs["input_ids"][0]):]
    code = processor.decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(code, encoding="utf-8")
    print(f"Wrote Ring 01 CadQuery output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/ring01_reference_images"))
    parser.add_argument("--output", type=Path, default=Path("data/ring01_output.py"))
    parser.add_argument("--checkpoint", default="maksimko123/cadrille")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    run(args.input_dir, args.output, args.checkpoint, args.max_new_tokens, args.device)


if __name__ == "__main__":
    main()
