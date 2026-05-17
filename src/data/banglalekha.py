"""BanglaLekha-Image-Captions dataset loader.

Source: Mendeley dataset rxxch9vw59 v2 (Mansoor et al., 2019).
  - `captions.json` is a list of {filename, caption: [str, str]} dicts;
    9,154 entries, two native-Bangla annotations per image.
  - `images/` is a flat directory of PNGs (filenames like `1.png` ... `9154.png`).

The dataset is single-split upstream; we partition deterministically by image
index using a 90/10 train/val split (seedable via the constructor).

Caption-multiplicity policy: each image yields *one* caption per `__getitem__`
call, chosen by `caption_selection`:
  - "first"  : always use captions[0]               (deterministic)
  - "random" : pick uniformly from the two captions (per-call, RNG-seeded)

For the slice / pipeline-validation run we default to "first" so a fixed seed
yields identical batches.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class BanglaLekhaItem:
    image: Image.Image
    caption: str
    filename: str


class BanglaLekhaCaptions(Dataset):
    """Real Bangla image-caption pairs from BanglaLekha-Image-Captions.

    Parameters
    ----------
    captions_path : Path to captions.json (the Mendeley file, untouched).
    images_dir    : Path to the extracted images/ directory.
    split         : "train", "val", or "all". Default "train".
    val_fraction  : Fraction of the corpus reserved for "val". Default 0.1.
    max_samples   : Cap dataset size after split (useful for slice runs).
    caption_selection : "first" or "random".
    seed          : Seed for the train/val partition (and for "random" selection).

    Notes
    -----
    Images are loaded lazily in __getitem__ so the dataset object stays light.
    `Image.open` keeps a file handle until the image is materialized; we copy
    to RGB and close immediately to avoid leaking handles across DataLoader
    workers.
    """

    def __init__(
        self,
        captions_path: str | Path,
        images_dir: str | Path,
        *,
        split: Literal["train", "val", "all"] = "train",
        val_fraction: float = 0.1,
        max_samples: int | None = None,
        caption_selection: Literal["first", "random"] = "first",
        seed: int = 42,
    ) -> None:
        self.captions_path = Path(captions_path)
        self.images_dir = Path(images_dir)
        self.caption_selection = caption_selection
        self._rng = random.Random(seed)

        with self.captions_path.open(encoding="utf-8") as f:
            entries = json.load(f)
        # Defensive: every entry must have at least one caption and an existing image.
        entries = [e for e in entries if e.get("caption") and (self.images_dir / e["filename"]).exists()]

        # Deterministic split by shuffling indices with the seed.
        idx = list(range(len(entries)))
        random.Random(seed).shuffle(idx)
        n_val = int(len(idx) * val_fraction)
        if split == "val":
            keep = set(idx[:n_val])
        elif split == "train":
            keep = set(idx[n_val:])
        elif split == "all":
            keep = set(idx)
        else:
            raise ValueError(f"unknown split: {split!r}")
        self.entries = [entries[i] for i in sorted(keep)]

        if max_samples is not None:
            self.entries = self.entries[:max_samples]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> BanglaLekhaItem:
        entry = self.entries[idx]
        captions = entry["caption"]
        if self.caption_selection == "first" or len(captions) == 1:
            caption = captions[0]
        else:
            caption = self._rng.choice(captions)

        path = self.images_dir / entry["filename"]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return BanglaLekhaItem(image=image, caption=caption.strip(), filename=entry["filename"])
