# runtime/execution — Phase 1 (CPU) ✅ / Phase 2 (KV cache) ✅ / Phase 3 (CUDA)

- `cpu_backend.h/.cpp`:
  - `forward(model, ids)` — Phase 1 baseline: drives embedding -> N transformer blocks
    -> final norm -> LM head, returning every intermediate (not just logits) so
    `tests/model_test.cpp` can check each one against `tests/golden/`. No KV cache:
    every call recomputes attention/MLP over the full sequence.
  - `prefill(model, ids, kv)` — Phase 2: identical computation to `forward()`, but also
    seeds `kv` with every layer's K/V for `ids` (typically the prompt).
  - `decode_step(model, token_id, position, kv)` — Phase 2: runs exactly one new token
    through the model using `kv`'s existing history, growing it by one row. Returns
    just that token's logits.

  Measured speedup from using the cache (`runtime/cache/README.md` has the full
  breakdown): **24.6x at 50 generated tokens, 72.3x at 200** — and the gap widens with
  length, since the uncached path's dominant cost (redoing every linear/MLP layer for
  every position, every step) is `O(N^2)` while the cached path's is `O(N)`.

Still to come:

- `cuda_backend` — dispatches the CUDA-kernel equivalents from `cuda/kernels`
  (Phase 3+). Only built when `BUILD_CUDA=ON`; this machine has no NVIDIA GPU/CUDA
  Toolkit (see `docs/architecture.md` §10), so Phase 3 needs a different machine or a
  cloud GPU instance.
- `kernel_registry` — maps (op, dtype, device) -> concrete kernel implementation, so
  the scheduler's backend/precision decisions (spec §19-20) don't need `if/else`
  chains scattered through the runtime. Not needed yet with only one backend (CPU).
