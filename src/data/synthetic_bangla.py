"""Tiny in-memory Bangla caption dataset for smoke testing.

Generates random-color PIL images paired with hand-written Bangla captions.
Used to verify training pipelines run end-to-end before real data is wired in.

DO NOT use this for any quantitative claim — images and captions are unrelated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset


# Hand-written Bangla captions covering common caption tropes
# (humans, objects, scenes, animals, actions). 16 distinct sentences.
BANGLA_CAPTIONS: list[str] = [
    "একটি লাল গাড়ি রাস্তায় চলছে।",
    "একজন মানুষ পার্কে হাঁটছে।",
    "একটি কালো কুকুর মাঠে দৌড়াচ্ছে।",
    "শিশুরা সমুদ্রতীরে খেলছে।",
    "একটি বিড়াল জানালার পাশে বসে আছে।",
    "একজন মহিলা বই পড়ছেন।",
    "নদীর উপর একটি নৌকা ভাসছে।",
    "পাহাড়ের চূড়ায় বরফ পড়েছে।",
    "একটি হলুদ ফুল বাগানে ফুটেছে।",
    "একজন কৃষক ধান কাটছেন।",
    "ছেলেটি সাইকেল চালাচ্ছে।",
    "মেয়েটি গান গাইছে।",
    "আকাশে অনেক পাখি উড়ছে।",
    "একটি ট্রেন স্টেশনে এসেছে।",
    "ছাত্ররা ক্লাসরুমে পড়ছে।",
    "চাঁদ আকাশে উজ্জ্বল হয়ে আছে।",
]


@dataclass
class SyntheticItem:
    image: Image.Image
    caption: str


def _make_image(seed: int, size: int = 224) -> Image.Image:
    """Random solid-color RGB image. Deterministic from seed."""
    rng = random.Random(seed)
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    return Image.new("RGB", (size, size), color)


class SyntheticBanglaCaptions(Dataset):
    """Each item: solid-color image + a Bangla sentence.

    Length is len(BANGLA_CAPTIONS) (16). Order is stable; image color is
    deterministic per index, so re-runs produce identical batches.
    """

    def __init__(self, image_size: int = 224) -> None:
        self.captions = BANGLA_CAPTIONS
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> SyntheticItem:
        return SyntheticItem(
            image=_make_image(seed=idx, size=self.image_size),
            caption=self.captions[idx],
        )
