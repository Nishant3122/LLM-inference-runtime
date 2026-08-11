# tests/

Test categories (spec §25): unit, integration, numerical, regression, performance,
stress. Phase 0 delivers the pattern (`tensor_test.cpp` + `tests/golden/`); the rest
of this list fills in as each phase lands:

| File | Phase | Status |
|---|---|---|
| `tensor_test.cpp` | 1 | ✅ written, not yet build-verified (no local toolchain, see `docs/architecture.md` §10) |
| `matmul_test.cpp` | 1 | ⬜ |
| `softmax_test.cpp` | 1 | ⬜ |
| `attention_test.cpp` | 1 | ⬜ — compare against `tests/golden/*/block_*_output.npy` |
| `tokenizer_test.cpp` | 1 | ⬜ |
| `model_test.cpp` | 1 | ⬜ — full forward pass vs. `tests/golden/*/logits.npy` |
| `kv_cache_test.cpp` | 2 | ⬜ — cached vs. recomputed logits must match (spec §32 ablation) |
| `quantization_test.cpp` | 5 | ⬜ |
| `runtime_test.cpp` | 6+ | ⬜ — end-to-end batched/adaptive runtime |

## `tests/golden/`

Reference outputs dumped from the trained PyTorch model by
[`reference/dump_golden.py`](../reference/dump_golden.py): fixed prompts -> exact
intermediate activations and logits, saved as `.npy`. This is the correctness oracle
for G1 (`docs/architecture.md` §4) — every C++ runtime path is compared against these,
not against "looks reasonable". See `tests/golden/manifest.json` for the prompt list,
token ids, and expected greedy next-token.

Regenerate after any change to `reference/model.py` or a retraining run:

```bash
cd reference && python dump_golden.py
```

## Numerical tolerance

FP32 CPU vs. the PyTorch reference should match near bit-for-bit (both are FP32,
same math, different implementation — expect ~1e-5 relative error from summation
order differences). FP16/INT8/INT4 are expected to diverge; track divergence with
absolute/relative error, cosine similarity, and generated-token agreement rather than
requiring exact equality (spec §24).
