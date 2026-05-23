"""BAN-Cap dataset loader.

Source: Khan et al. (LREC 2022), arXiv:2205.14462. Distributed as a single
CSV via Kaggle (https://www.kaggle.com/datasets/saifsust/bancap).

  - `BAN-Cap_captiondata.csv` has 40,455 rows with columns
    {caption_id, english_caption, bengali_caption}. `caption_id` is
    `<image_filename>#<index>` with index 0..4. Every image has exactly
    five Bengali captions, one per annotator.
  - Images come from the Flickr8k corpus (8,091 .jpg files). They are
    NOT bundled with the captions; pull from a Flickr8k mirror
    (e.g. Kaggle's `adityajn105/flickr8k`).

Compared to BanglaLekha-Image-Captions, BAN-Cap has:
  - 5 refs/image vs. 2 (and the 5 are peer captions from different
    annotators rather than short-summary + long-description pairs)
  - 3× the unique Bengali token count (16,560 vs. 5,377)
  - A much flatter caption-ending distribution: the most common
    sentence-final word (`আছে`) covers 13.5% of captions in BAN-Cap
    vs. 46.4% in BanglaLekha — i.e. weaker corpus prior

The dataset is single-split upstream; we partition deterministically by
image index using a 90/10 train/val split (seedable via the constructor).

Caption-multiplicity policy: each image yields *one* caption per
`__getitem__` call, chosen by `caption_selection`:
  - "first"    : always use captions[0]                  (deterministic)
  - "last"     : always use captions[-1]                 (deterministic)
  - "random"   : pick uniformly from the 5 captions      (per-call, RNG-seeded)
  - "index:<i>": always use captions[i] for i in 0..4    (deterministic)

For BAN-Cap the 5 captions are peers from different annotators, not
short/long pairs. "first" and "last" therefore vary by annotator
identity rather than by caption length.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class BanCapItem:
    image: Image.Image
    caption: str
    filename: str
    # All 5 references available for the item — useful for multi-ref
    # scoring at eval time (BLEU/CIDEr/BERTScore). The chosen `caption`
    # is one of these.
    references: list[str]


class BanCapCaptions(Dataset):
    """BAN-Cap image-caption pairs (Bengali side).

    Parameters
    ----------
    captions_path : Path to BAN-Cap_captiondata.csv (the Kaggle file, untouched).
    images_dir    : Path to a directory of Flickr8k images (flat .jpg).
    split         : "train", "val", or "all". Default "train".
    val_fraction  : Fraction of images reserved for "val". Default 0.1.
    max_samples   : Cap dataset size after split (useful for slice runs).
    caption_selection : "first", "last", "random", or "index:<N>" for N in 0..4.
    seed          : Seed for the train/val partition and for "random" selection.

    Notes
    -----
    Images are loaded lazily in __getitem__ so the dataset object stays light.
    """

    def __init__(
        self,
        captions_path: str | Path,
        images_dir: str | Path,
        *,
        split: Literal["train", "val", "all"] = "train",
        val_fraction: float = 0.1,
        max_samples: int | None = None,
        caption_selection: str = "first",
        seed: int = 42,
    ) -> None:
        self.captions_path = Path(captions_path)
        self.images_dir = Path(images_dir)
        self.caption_selection = caption_selection
        self._rng = random.Random(seed)
        self._parsed_index = self._parse_index_selector(caption_selection)

        # Read all rows, group by image filename.
        per_image: dict[str, list[tuple[int, str]]] = defaultdict(list)
        with self.captions_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cap_id = row["caption_id"]
                fname, idx_str = cap_id.split("#")
                per_image[fname].append((int(idx_str), row["bengali_caption"].strip()))

        # Order each group by the explicit index (CSV order should already
        # be sorted, but rely on the index to be defensive).
        entries: list[dict] = []
        for fname, pairs in per_image.items():
            pairs.sort(key=lambda p: p[0])
            captions = [c for _, c in pairs]
            if not captions:
                continue
            if not (self.images_dir / fname).exists():
                continue
            entries.append({"filename": fname, "captions": captions})

        # Deterministic train/val split by shuffled image index.
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

    @staticmethod
    def _parse_index_selector(sel: str) -> int | None:
        """Return the integer N for selectors of the form 'index:N', else None."""
        if not sel.startswith("index:"):
            return None
        try:
            n = int(sel.split(":", 1)[1])
        except ValueError as e:
            raise ValueError(f"invalid index selector {sel!r}: {e}")
        if not (0 <= n <= 4):
            raise ValueError(f"index selector out of range 0..4: {sel!r}")
        return n

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> BanCapItem:
        entry = self.entries[idx]
        captions: list[str] = entry["captions"]

        sel = self.caption_selection
        if self._parsed_index is not None:
            i = self._parsed_index
            caption = captions[i] if i < len(captions) else captions[-1]
        elif sel == "first" or len(captions) == 1:
            caption = captions[0]
        elif sel == "last":
            caption = captions[-1]
        elif sel == "random":
            caption = self._rng.choice(captions)
        else:
            raise ValueError(f"unknown caption_selection: {sel!r}")

        path = self.images_dir / entry["filename"]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return BanCapItem(
            image=image,
            caption=caption,
            filename=entry["filename"],
            references=list(captions),
        )
