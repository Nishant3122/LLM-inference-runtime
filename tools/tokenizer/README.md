# tools/tokenizer — Phase 1 ✅

C++ port of [`reference/tokenizer.py::CharTokenizer`](../../reference/tokenizer.py):
`tok::CharTokenizer::load("models/vocab.bin")`, then `.encode(text)` /
`.decode(ids)`. Loads the binary vocab format (`docs/model_format.md`), not
`vocab.json` — no JSON parser needed for one small fixed-shape file. No tokenizer
*training* is in scope (spec §5 non-goals) — the vocab is fixed at export time from
the character set the reference model was trained on.

See `tests/tokenizer_test.cpp` for a correctness check against
`tests/golden/manifest.json`'s known prompt -> id mappings.
