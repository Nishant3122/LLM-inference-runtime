# tests/

Test categories (spec §25): unit, integration, numerical, regression, performance,
stress. Phase 0/1 deliver the pattern (`tensor_test.cpp`, `tokenizer_test.cpp`,
`model_test.cpp` + `tests/golden/`); the rest of this list fills in as each phase lands:

| File | Phase | Status |
|---|---|---|
| `tensor_test.cpp` | 1 | ✅ passing — `runtime/core` (Shape/Tensor/dtype) |
| `tokenizer_test.cpp` | 1 | ✅ passing — `tools/tokenizer` vs. `tests/golden/manifest.json` ids |
| `model_test.cpp` | 1 | ✅ passing — full forward pass (embedding, every block, final norm, logits) vs. `tests/golden/*.bin`, max abs diff ~1e-6 |
| `kv_cache_test.cpp` | 2 | ✅ passing — cached (`prefill`+`decode_step`) vs. recomputed (`forward`) logits match with **max abs diff = 0** at every step (spec §32 ablation) |
| `quantization_test.cpp` | 5 | ⬜ |
| `runtime_test.cpp` | 6+ | ⬜ — end-to-end batched/adaptive runtime |

No separate `matmul_test.cpp`/`softmax_test.cpp`/`attention_test.cpp`: those ops
(`runtime/transformer/ops.cpp`, `attention.cpp`) are exercised and checked
transitively through `model_test.cpp`'s per-block golden comparison, which already
bisects a mismatch to a specific block. Split them out only if a future change needs
to test one of those ops in isolation (e.g. before a Phase 4 optimization changes
`linear`'s internals) — Engineering Principle 4.

## `tests/golden/`

Reference outputs dumped from the trained PyTorch model by
[`reference/dump_golden.py`](../reference/dump_golden.py): fixed prompts -> exact
intermediate activations and logits, saved twice — as `.npy` (Python-side inspection)
and as `.bin` (a minimal `[rows][cols][float32 data]` format `model_test.cpp` reads
directly, no npy-header or JSON parser needed in C++). This is the correctness oracle
for G1 (`docs/architecture.md` §4) — every C++ runtime path is compared against these,
not against "looks reasonable". See `tests/golden/manifest.json` for the prompt list,
token ids, and expected greedy next-token (the exact input-id arrays are also copied
as constants into `tests/model_test.cpp` and `tests/tokenizer_test.cpp`).

Regenerate after any change to `reference/model.py` or a retraining run:

```bash
cd reference && python dump_golden.py
```

## Numerical tolerance

FP32 CPU vs. the PyTorch reference matches near bit-for-bit (both are FP32, same
math, different implementation — summation-order differences only). Predicted ~1e-5;
observed in `model_test.cpp` on the Stage-1 model: max abs diff ~1e-6 to 1e-7 on
intermediate activations, ~5e-6 on final logits (error compounds slightly over 4
layers + softmax/GELU nonlinearities, as expected). Test tolerance is set to 1e-3 —
~1000x margin above observed noise, still tight enough to catch a real bug.
FP16/INT8/INT4 (Phase 5) are expected to diverge; track divergence with
absolute/relative error, cosine similarity, and generated-token agreement rather than
requiring exact equality (spec §24).
