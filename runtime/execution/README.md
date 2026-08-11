# runtime/execution — Phase 1 (CPU) / Phase 3 (CUDA)

Not yet implemented. Will contain:

- `cpu_backend` — dispatches `runtime/transformer` ops on CPU (Phase 1).
- `cuda_backend` — dispatches the CUDA-kernel equivalents from `cuda/kernels`
  (Phase 3+). Only built when `BUILD_CUDA=ON`; this machine currently has no NVIDIA
  GPU/CUDA Toolkit (see `docs/architecture.md` §10), so Phase 3 needs a different
  machine or a cloud GPU instance.
- `kernel_registry` — maps (op, dtype, device) -> concrete kernel implementation, so
  the scheduler's backend/precision decisions (spec §19-20) don't need `if/else`
  chains scattered through the runtime.
