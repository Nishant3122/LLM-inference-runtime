# runtime/model — Phase 1

Not yet implemented. Will contain:

- `model_loader` — reads `model.bin` per [`docs/model_format.md`](../../docs/model_format.md)
  (header + tensor table + data section) and produces in-memory `rt::Tensor` views.
- `weight_manager` — owns the loaded weight buffers (the one place in Phase 1 that
  actually allocates/frees memory for model weights).
- `model_config` — mirrors `reference/model.py::TinyTransformerConfig` fields
  (`vocab_size`, `d_model`, `n_layers`, `n_heads`, `d_ff`, `context_length`), read
  from the `model.bin` header.

Reference for exact byte layout and tensor names: [`docs/model_format.md`](../../docs/model_format.md),
and the Python side that must match it, [`reference/export.py`](../../reference/export.py).
