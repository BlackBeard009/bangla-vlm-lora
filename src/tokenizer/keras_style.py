"""Minimal reimplementation of Keras's ``Tokenizer`` word-level behavior.

Needed for the Bornon reproduction. The paper tokenizes captions with "Keras's
text tokenizer" and a top-5,000-word vocabulary. The protocol-relevant
quirks reproduced here, matching tf.keras.preprocessing.text.Tokenizer:

  - fit: split on whitespace after stripping the default ASCII filter
    characters; rank words by frequency (ties broken by first-seen
    order); index 0 is reserved for padding.
  - texts_to_sequences with ``num_words=N``: words with index >= N are
    silently DROPPED from the sequence — no OOV token, no placeholder.
    This is the quirk that matters for BLEU: rare words vanish from
    both hypotheses and (if references are round-tripped through the
    tokenizer) from references too.

The default Keras filter string is ASCII-only, so the Bangla danda
(U+0964) and all Bangla codepoints pass through untouched.
"""

from __future__ import annotations

from collections import Counter

_KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


def _keras_split(text: str) -> list[str]:
    table = str.maketrans({c: " " for c in _KERAS_FILTERS})
    return text.lower().translate(table).split()


class KerasStyleTokenizer:
    def __init__(self, num_words: int | None = None) -> None:
        self.num_words = num_words
        self.word_index: dict[str, int] = {}

    def fit_on_texts(self, texts: list[str]) -> None:
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for text in texts:
            for word in _keras_split(text):
                counts[word] += 1
                first_seen.setdefault(word, len(first_seen))
        ranked = sorted(
            counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]])
        )
        # Index 0 reserved for padding, as in Keras.
        self.word_index = {w: i + 1 for i, (w, _) in enumerate(ranked)}

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        limit = self.num_words
        out = []
        for text in texts:
            seq = []
            for word in _keras_split(text):
                idx = self.word_index.get(word)
                if idx is None:
                    continue
                if limit is not None and idx >= limit:
                    continue  # silent drop — the Keras num_words quirk
                seq.append(idx)
            out.append(seq)
        return out

    def sequences_to_texts(self, sequences: list[list[int]]) -> list[str]:
        inverse = {i: w for w, i in self.word_index.items()}
        return [" ".join(inverse[i] for i in seq if i in inverse)
                for seq in sequences]
