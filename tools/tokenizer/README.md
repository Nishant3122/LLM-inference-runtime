# tools/tokenizer — Phase 1

Not yet implemented. Will be a small C++ port of
[`reference/tokenizer.py::CharTokenizer`](../../reference/tokenizer.py): load
`models/vocab.json`, encode/decode. No tokenizer *training* is in scope (spec §5
non-goals) — the vocab is fixed at export time from the character set the reference
model was trained on.
