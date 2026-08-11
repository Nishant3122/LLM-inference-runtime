# runtime/model — Phase 1 ✅

- `model_config.h` — mirrors `reference/model.py::TinyTransformerConfig`.
- `model.h` — `Model`: owns the raw weight bytes read from `model.bin` and exposes
  named `rt::Tensor` views into them (`tok_embedding`, `pos_embedding`, one
  `transformer::BlockWeights` per layer, `ln_f_*`, `lm_head_*`).
- `model_loader.h/.cpp` — parses `model.bin` per
  [`docs/model_format.md`](../../docs/model_format.md) (header + tensor table + data
  section) into a `Model`.

**Deviation from the original spec (documented per §45, "what can be changed"):**
there's no separate `weight_manager` — `Model` owns its storage directly. With one
load path and one owner, splitting them would just be `Model` with extra indirection
(Engineering Principle 4). Revisit if/when a CUDA-resident weight story needs
different ownership than the CPU path.

Validated against `tests/golden/` in `tests/model_test.cpp` — see
`docs/architecture.md` §4 (G1).
