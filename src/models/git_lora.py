"""LoRA wiring for GiT.

GiT has two distinct attention naming conventions:

- CLIP vision encoder side: ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj``
- Text transformer side:    ``query`` / ``key`` / ``value``

For Bangla adaptation we want to adapt the language pathway, so by default
we target the text-side projections. CLIP-side adaptation is exposed for
ablations.

Known concern (HF peft discussions #1958): passing ``task_type=CAUSAL_LM``
to ``LoraConfig`` triggers wrappers that mis-fire on GiT and the adapter
fails to learn. We deliberately do NOT set ``task_type`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from peft import LoraConfig, get_peft_model

from .git import GitBundle, load_git


# Text-side projections. These exist in `git.encoder.layer.{i}.attention.self.{query,key,value}`.
GIT_TEXT_TARGETS: tuple[str, ...] = ("query", "key", "value")
# CLIP-side projections — exposed for ablation. Not used by default.
GIT_VISION_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


def apply_lora(
    bundle: GitBundle,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = GIT_TEXT_TARGETS,
) -> GitBundle:
    """Wrap `bundle.model` in a PEFT LoRA adapter.

    Returns a new bundle with the wrapped model. The processor is unchanged.
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        # Intentionally no task_type — see module docstring.
    )
    lora_model = get_peft_model(bundle.model, config)
    return GitBundle(model=lora_model, processor=bundle.processor, name=bundle.name)


def trainable_param_summary(model: torch.nn.Module) -> dict:
    """Return a small dict for logging; avoids dependency on PEFT internals."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
