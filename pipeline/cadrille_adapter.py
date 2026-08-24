"""Optional adapter boundary for Cadrille reconstruction backends.

Importing this module does not import PyTorch or Transformers.  Those heavy
dependencies are loaded only when the adapter is instantiated, allowing the
pipeline contracts and non-ML stages to run in a clean environment.
"""

from __future__ import annotations

from typing import Any

from .schema import ComponentProposal, ReconstructionArtifact


class CadrilleAdapter:
    """Bridge a foundation-model backend into the shared artifact format."""

    def __init__(self, checkpoint: str, device: str = "auto") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor
            from cadrille import Cadrille
        except ImportError as exc:
            raise RuntimeError(
                "Cadrille dependencies are not installed. Install the ML "
                "environment before loading this adapter."
            ) from exc

        kwargs = {"torch_dtype": torch.bfloat16}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self.model = Cadrille.from_pretrained(self.checkpoint, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", padding_side="left"
        )

    def propose_components(
        self, artifact: ReconstructionArtifact
    ) -> ReconstructionArtifact:
        if self.model is None:
            self.load()
        raise NotImplementedError(
            "Inference wiring is dataset/checkpoint-specific. Implement this "
            "method after selecting the foundation-model input contract."
        )

