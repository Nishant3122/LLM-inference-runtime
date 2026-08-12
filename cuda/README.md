# cuda/ — Phase 3 ✅ (naive) / Phase 4 ✅ (profiled + fused)

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

## Phase 4: profile first, then fuse

`examples/benchmark.cpp` (CPU) and `examples/benchmark_cuda.cu` (CUDA) profile
`forward()`/`forward_cuda()` across `T={8,32,128,256}` using `runtime/profiling`
(Engineering Principle 1: profile before optimizing). The two backends showed
completely different bottlenecks for this tiny model:

| | CPU | CUDA |
|---|---|---|
| Dominant op | `mlp` (~60-67%) | `attention` (38-56%) + `layer_norm` (30-38%), **combined** |
| `mlp` share | 60-67% | only 12-17% |

CPU is a straightforward compute-bound profile (MLP does the most FLOPs, takes the
most time). CUDA is the opposite signature: `causal_self_attention` was **7 separate
kernel launches per call** (wq/wk/wv/scores/softmax/weighted-sum/wo), and
`layer_norm` gets called 9x per forward pass on tiny buffers — classic
launch-overhead dominance, not compute cost. (GPU still crushed CPU overall: ~54x
faster at T=256, only ~5x at T=8, since overhead eats a bigger fraction of tiny-T runs.)

This pointed at **kernel fusion** (fewer launches), not shared-memory tiling (which
helps compute-bound kernels — profiling showed this model isn't compute-bound on GPU
at this scale). Implemented, with the naive baseline kept fully intact for A/B
comparison (Engineering Principle 2):

- `cuda::ops::linear_fused` — linear + optional GELU epilogue + optional in-place
  residual-add epilogue, one kernel launch instead of up to three.
- `cuda::transformer::causal_self_attention_fused` — Q/K/V in one launch
  (`qkv_fused_kernel`) instead of three, and `wo`'s output accumulates directly into
  the residual stream instead of a separate `add_inplace` call. **7 launches → 5.**
- Fused MLP path: fc1+bias+GELU in one launch, fc2+residual in one launch.
  **4 launches (fc1, gelu, fc2, add) → 2.**
- `forward_cuda(..., use_fusion=true)` — opt-in flag, default `false` keeps the
  Phase 3 baseline unchanged.

**Correctness**: `attention_fused` vs. naive+manual-add matches **exactly** (diff=0);
full-model fused vs. naive matches **exactly** (diff=0) — fusion changed launch
count, not any computed value (`tests/cuda_ops_test.cu`).

**Performance** (unprofiled wall-clock — profiling itself forces a sync per timed
section, which distorts absolute numbers; wall-clock is what actually answers
whether fusion helped):

| T | naive | fused | speedup |
|---|---|---|---|
| 8 | 9.125ms | 7.898ms | 1.16x |
| 32 | 12.598ms | 9.460ms | **1.33x** |
| 128 | 21.575ms | 18.175ms | 1.19x |
| 256 | 33.897ms | 32.968ms | 1.03x |

Modest, not dramatic — and that's an honest, predictable result, not a shortfall:
speedup peaks at T=32 and nearly vanishes by T=256, exactly where the model predicts
it should. At T=256, attention's `O(T^2)` compute has grown large enough to genuinely
dominate over launch overhead, so removing 2 launches matters proportionally less.
Per Engineering Principle 2: "this optimization improves latency by 1.03x-1.33x
across T=8-256, largest at T=32" — not "fusion made it faster," full stop.

## What isn't here yet

- **No CUDA-side KV cache.** `forward_cuda()` still does full `O(T^2)` recompute
  every call — the CUDA equivalent of Phase 1's `cpu_backend::forward()` before
  Phase 2 added caching. Porting `runtime/cache` to CUDA (device-resident
  `LayerKVCache`, cached attention kernel) is natural follow-up work, not done in
  this pass. Given the KV-cache speedup on CPU was 24-72x (`runtime/cache/README.md`)
  vs. fusion's 1.03-1.33x here, this is very likely a much bigger win than any further
  kernel-level optimization — a good candidate for whatever comes after Phase 4.
- **No batching, no quantization.** Phases 5/6 respectively.
- **No shared-memory tiling.** Deliberately not attempted — profiling showed this
  model isn't compute-bound on GPU at this scale, so tiling wouldn't be addressing
  the actual bottleneck (Engineering Principle 1).
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
