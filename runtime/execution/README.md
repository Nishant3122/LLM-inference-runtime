# runtime/execution — Phase 1 (CPU) ✅ / Phase 3 (CUDA)

- `cpu_backend.h/.cpp` — `forward(model, ids)` drives embedding -> N transformer
  blocks -> final norm -> LM head, returning every intermediate (not just logits) so
  `tests/model_test.cpp` can check each one against `tests/golden/`. No KV cache: every
  call recomputes attention/MLP over the full sequence — this is the Phase 1 baseline
  Phase 2's with/without-cache ablation (spec §32) measures against. Ad hoc timing on
  this machine: 50-token generation from a 7-token prompt (Stage-1 config, `T` growing
  to 57) took ~2.4s; the quadratic-ish per-step cost (attention is `O(T^2)`, and every
  step recomputes from scratch) is expected to dominate until Phase 2.

Still to come:

- `cuda_backend` — dispatches the CUDA-kernel equivalents from `cuda/kernels`
  (Phase 3+). Only built when `BUILD_CUDA=ON`; this machine has no NVIDIA GPU/CUDA
  Toolkit (see `docs/architecture.md` §10), so Phase 3 needs a different machine or a
  cloud GPU instance.
- `kernel_registry` — maps (op, dtype, device) -> concrete kernel implementation, so
  the scheduler's backend/precision decisions (spec §19-20) don't need `if/else`
  chains scattered through the runtime. Not needed yet with only one backend (CPU).
