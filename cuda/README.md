# cuda/ — Phase 3 ✅ (naive)

Built and verified 2026-08-13 on a real NVIDIA Tesla T4 (Google Colab free tier, CUDA
Toolkit 12.8, driver 580.82.07) — this machine has no GPU (`docs/architecture.md`
§10), so all CUDA development happens locally (no compiler to check syntax against)
and gets synced to Colab to actually compile/run. See `scripts/colab_workflow.md`.

- `kernels/` — `linear` (matmul+bias), `softmax_rows`, `layer_norm`, `gelu`/`add`
  (elementwise), `embedding_forward`, `causal_self_attention`. Each mirrors its
  `runtime/transformer` CPU counterpart op-for-op — naive (one thread per output
  element, no tiling/shared-memory/fusion), matching spec §14's development order
  (naive CPU -> naive CUDA -> correct CUDA -> profile -> optimize). Optimization is
  Phase 4, after profiling says a specific kernel is the bottleneck.
- `memory/device_buffer.h` — RAII device buffer (move-only, one `cudaMalloc` per
  buffer). `memory/cuda_model.h` + `model_upload.cu` — uploads a loaded
  `model::Model`'s weight blob to device memory once (one `cudaMalloc` + one
  `cudaMemcpy` for everything) and rebuilds the same named `BlockWeights`/
  `AttentionWeights`/`MlpWeights` views with device pointers instead of host ones —
  reused unchanged from the CPU side, since `Tensor` already carries a `Device` tag.
- `utils/cuda_check.h` — `CUDA_CHECK`/`CUDA_CHECK_LAST_ERROR` macros; every CUDA
  runtime call and kernel launch in this project is wrapped in one of these.

## Correctness: every kernel vs. its CPU counterpart

`tests/cuda_ops_test.cu` (`-DBUILD_CUDA=ON`), random synthetic data, on the real T4:

| Op | max abs diff |
|---|---|
| linear | 5.96e-07 |
| softmax_rows | 5.96e-08 |
| layer_norm | 2.38e-07 |
| gelu | 2.38e-07 |
| embedding | **0** (exact) |
| causal_self_attention | 1.91e-06 |
| **full model forward** (`forward_cuda()` vs `forward()`, real trained weights) | 1.79e-06 |

Same range as the CPU-vs-PyTorch numbers in `tests/README.md` — FP32 rounding noise
from different summation order, not a correctness gap. `forward_cuda()`
(`runtime/execution/cuda_backend.h`) reuses `ForwardResult` from `cpu_backend.h`
unchanged, so it's diffed with the exact same comparison code `tests/model_test.cpp`
uses for the CPU path.

## What isn't here yet

- **No CUDA-side KV cache.** `forward_cuda()` is the Phase 3 baseline: full
  `O(T^2)` recompute every call, the CUDA equivalent of Phase 1's `cpu_backend::forward()`
  before Phase 2 added caching. Porting `runtime/cache` to CUDA (device-resident
  `LayerKVCache`, cached attention kernel) is natural follow-up work, not done in this
  pass — scoped out to keep Phase 3 focused on "naive CUDA is numerically correct"
  before adding more surface area.
- **No batching, no quantization, no kernel fusion.** Phases 5/6/4 respectively.
- **`CMAKE_CUDA_ARCHITECTURES` defaults to 75** (Turing/T4). Override with
  `-DCMAKE_CUDA_ARCHITECTURES=...` for a different GPU (e.g. 86 for Ampere consumer
  cards, 80 for A100) — untested on anything but a T4 so far.

## Real bugs found getting here (for the record)

Three, all caught by the Colab round-trip since there's no local compiler to catch
them earlier:
1. `cuda_model.h` used `model::Model` but only included `model_config.h`, not
   `model.h` (where `Model` is declared) — `namespace "rt::model" has no member "Model"`.
2. `tests/cuda_ops_test.cu` called `rt::transformer::embedding_forward` without
   including `runtime/transformer/embedding.h`.
3. `CUDA_SEPARABLE_COMPILATION` (relocatable device code) was turned on for
   `cuda_kernels` needlessly — nothing calls a `__device__` function across `.cu`
   files — and broke linking for every consumer with
   `undefined reference to __cudaRegisterLinkedBinary_...`. Removed.

`CMakeLists.txt` at the repo root gates all of this behind `-DBUILD_CUDA=ON` (off by
default) and calls `enable_language(CUDA)` only when that flag is set, so a CPU-only
checkout still configures cleanly. `cuda/` is configured *before* `runtime/` in the
top-level `CMakeLists.txt` specifically so `runtime` can link against the
`cuda_kernels` target when `BUILD_CUDA=ON`.
