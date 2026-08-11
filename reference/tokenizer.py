"""Minimal character-level tokenizer.

Deliberately trivial: the runtime spec (docs/architecture.md) treats tokenization as
orthogonal to the model runtime, and char-level vocab keeps Stage 1 free of BPE
complexity. `tools/tokenizer/` will eventually host a C++ port of just the
encode/decode table lookup this class does (no training needed at inference time,
since the vocab is baked into `vocab.json` at export time).
"""
import json
from pathlib import Path
from typing import List


class CharTokenizer:
    def __init__(self, chars: List[str]):
        self.chars = list(chars)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def from_json(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data["id_to_char"]
        return cls(chars)

    def save(self, path: str) -> None:
        data = {
            "char_to_id": self.char_to_id,
            "id_to_char": [self.id_to_char[i] for i in range(self.vocab_size)],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def encode(self, text: str) -> List[int]:
        try:
            return [self.char_to_id[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} not in vocab") from e

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)
