# runtime/profiling — Phase 4 ✅ (minimal)

`profiler.h`/`.cpp`: a label -> `{calls, total_ms}` accumulator with a
`time(label, fn)` helper, used to answer spec §29's Q7 ("which Transformer
operations dominate execution time?") before deciding what to optimize
(Engineering Principle 1: profile first).

Deliberately not CUDA-aware itself — no `cuda_runtime.h` dependency, so it builds
without `BUILD_CUDA` and works for CPU timing too. For GPU work, the *caller* is
responsible for synchronizing inside the timed lambda (kernel launches are async;
see `runtime/execution/cuda_backend.cu` for the pattern) — the profiler can't know
when a kernel actually finished on its own.

Opt-in only: every instrumented call site (`cpu_backend::forward()`,
`cuda_backend::forward_cuda()`) takes an optional `Profiler*` defaulting to
`nullptr`, so normal (uninstrumented) calls pay zero overhead — no `std::function`
wrapping, no timer calls, nothing. `examples/benchmark.cpp` (CPU) and
`examples/benchmark_cuda.cu` (GPU, `BUILD_CUDA`-gated) are the two places that
actually pass a profiler in.

Coarse categories for the first profiling pass (not yet split further — spec §30
also wants QKV projection/softmax broken out separately from "attention" as a
whole; do that only if the coarse pass shows attention dominating and the finer
breakdown is actually needed to decide what to optimize):

- `embedding`, `layer_norm` (all LN calls combined), `attention` (whole
  `causal_self_attention` call), `mlp` (fc1+gelu+fc2), `residual_add`, `lm_head`.

Not a general-purpose profiler (no `metrics`/`trace` submodules from the original
spec's `profiling/` directory listing) — built specifically to answer "what should
Phase 4 optimize," nothing more, per Engineering Principle 4.
