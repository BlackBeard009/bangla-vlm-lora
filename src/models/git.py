"""GiT (Generative Image-to-text Transformer) model loading helpers.

Wraps `transformers.AutoProcessor` + `AutoModelForCausalLM` for
`microsoft/git-*` checkpoints. Kept dumb on purpose — LoRA wiring goes
elsewhere so the base loader stays useful for full-fine-tune baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BertTokenizer


@dataclass
class GitBundle:
    model: torch.nn.Module
    processor: object  # AutoProcessor, no clean type
    name: str


def load_git(
    name: str = "microsoft/git-base",
    dtype: torch.dtype = torch.float32,
    use_fast_tokenizer: bool = True,
) -> GitBundle:
    """Load a GiT model + processor.

    Set use_fast_tokenizer=False before any large vocabulary extension —
    the Rust `tokenizers` AddedVocabulary path panics ("bad split") when
    many script-foreign tokens are added in batch. The slow tokenizer is
    slower at encode-time but correct, and tokenization is rarely the
    bottleneck (the GPU is).
    """
    processor = AutoProcessor.from_pretrained(name)
    if not use_fast_tokenizer:
        processor.tokenizer = BertTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return GitBundle(model=model, processor=processor, name=name)
