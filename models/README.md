# models/

Generated artifacts, gitignored (see root `.gitignore`) — all reproducible from
`reference/`:

| File | Produced by | Format |
|---|---|---|
| `tiny_transformer.pt` | `reference/train.py` | PyTorch checkpoint (state dict + config) |
| `vocab.json` | `reference/train.py` | char-level vocab, see `reference/tokenizer.py` |
| `model.bin` | `reference/export.py` | custom binary format, see `docs/model_format.md` — this is what the C++ runtime loads |

Regenerate from scratch:

```bash
cd reference
python data/make_corpus.py
python train.py
python export.py
```
