"""Minimal character-level tokenizer.

Deliberately trivial: the runtime spec (docs/architecture.md) treats tokenization as
orthogonal to the model runtime, and char-level vocab keeps Stage 1 free of BPE
complexity. `tools/tokenizer/` hosts the C++ port of just the encode/decode table
lookup this class does (no training needed at inference time, since the vocab is
baked in at export time).

Two on-disk forms:
  - `vocab.json` (`save`/`from_json`): human-readable, used by the Python side.
  - `vocab.bin` (`save_binary`): what the C++ runtime actually loads, since writing a
    JSON parser in C++ for one small fixed-shape file isn't worth it (Engineering
    Principle 4). See docs/model_format.md "Companion file: vocab.bin" for the layout.
"""
import json
import struct
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

    def save_binary(self, path: str) -> None:
        """vocab.bin layout (little-endian), see docs/model_format.md:
        [magic uint32 "TVOC"][version uint32][vocab_size uint32]
        then, for id in [0, vocab_size): [utf8_byte_len uint32][utf8 bytes]
        (char-level, but stored as UTF-8 byte strings so non-ASCII corpora still work).
        """
        MAGIC = 0x54564F43  # "TVOC" little-endian
        VERSION = 1
        with open(path, "wb") as f:
            f.write(struct.pack("<III", MAGIC, VERSION, self.vocab_size))
            for i in range(self.vocab_size):
                b = self.id_to_char[i].encode("utf-8")
                f.write(struct.pack("<I", len(b)))
                f.write(b)

    def encode(self, text: str) -> List[int]:
        try:
            return [self.char_to_id[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} not in vocab") from e

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)
