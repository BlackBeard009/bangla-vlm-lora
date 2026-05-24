"""Dataset registry for the bangla-vlm-lora project.

A thin dispatch over the per-corpus loaders. Scripts that don't care
which corpus they target (diagnostics, scoring, future eval scripts)
read the loader name from config and stay corpus-agnostic.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from .bancap import BanCapCaptions
from .banglalekha import BanglaLekhaCaptions

__all__ = ["BanglaLekhaCaptions", "BanCapCaptions", "build_caption_dataset"]


def build_caption_dataset(loader: str, **kwargs) -> Dataset:
    """Construct a caption dataset by name.

    Supported loader names: ``banglalekha``, ``bancap``. Keyword arguments
    are forwarded unchanged to the chosen loader's constructor — both
    accept the same core kwargs (``captions_path``, ``images_dir``,
    ``split``, ``val_fraction``, ``max_samples``, ``caption_selection``,
    ``seed``); ``BanCapCaptions`` additionally supports ``caption_selection="index:N"``.
    """
    if loader == "banglalekha":
        return BanglaLekhaCaptions(**kwargs)
    if loader == "bancap":
        return BanCapCaptions(**kwargs)
    raise ValueError(
        f"unknown data loader: {loader!r}. Expected 'banglalekha' or 'bancap'."
    )
