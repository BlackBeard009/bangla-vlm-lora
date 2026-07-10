"""Reload a saved bridged-GiT + LoRA checkpoint for inference.

The save layout (produced by the training scripts with
``output.save_adapter: true``) is:

  adapter_model.safetensors   LoRA deltas + modules_to_save weights
  adapter_config.json
  tokenizer.json + tokenizer_config.json   (bridged tokenizer)

Restore order matters: load base GiT → restore bridged tokenizer →
resize base embeddings to the bridged vocab → load the PEFT adapter
(which restores both LoRA deltas and the trained word_embeddings /
output head via modules_to_save). This is the pattern validated in
scripts/decode_ablation.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def load_bridged_lora(
    base_name: str,
    adapter_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
):
    """Return ``(peft_model, processor)`` ready for eval-mode generation."""
    processor = AutoProcessor.from_pretrained(base_name)
    bridged_tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    processor.tokenizer = bridged_tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    base_model.resize_token_embeddings(len(bridged_tokenizer))

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.to(device)
    peft_model.eval()
    return peft_model, processor
