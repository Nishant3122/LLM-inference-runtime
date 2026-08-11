# cuda/ — Phase 3+

Not yet implemented, and not buildable on the primary dev machine as of Phase 0: no
NVIDIA GPU/driver and no CUDA Toolkit are installed (`docs/architecture.md` §10).
Building this out needs either installing the CUDA Toolkit on a machine with an
NVIDIA GPU, or using a cloud GPU instance.

Planned contents (spec §14-15):

- `kernels/` — matmul, vector ops, normalization, softmax, attention, quantized ops.
  Development order: naive CPU (done in `runtime/transformer`) -> naive CUDA -> correct
  CUDA -> profile -> optimize bottlenecks only (don't hand-optimize ops that profiling
  doesn't flag).
- `memory/` — the CUDA memory pool (spec §13): avoid `cudaMalloc`/`cudaFree` during
  token generation; persistent buffers for weights, KV cache, activations, workspace.
- `utils/` — CUDA error-checking macros, device query helpers, stream/event wrappers.

`CMakeLists.txt` at the repo root already gates this behind `-DBUILD_CUDA=ON` (off by
default) and calls `enable_language(CUDA)` only when that flag is set, so a CPU-only
checkout of this repo still configures cleanly.
