# Benchmarking

Mandatory per `docs/architecture.md` Engineering Principle 2: every optimization
needs a measured baseline. This document defines what gets measured and how, ahead of
there being anything to benchmark yet (Phase 0) — so `tools/benchmark` has a spec to
implement against instead of improvising one later.

## Metrics (spec §27)

| Metric | Definition |
|---|---|
| TTFT | Time to first token — prompt/prefill processing latency |
| TPOT | Time per output token — average latency of each subsequent forward pass |
| Throughput | tokens/sec, aggregate across a batch |
| End-to-end latency | request received -> complete response |
| Memory | model weights, KV-cache, peak GPU/CPU memory, separately |

## Matrix (spec §28)

Vary, independently:

- **Batch size:** 1, 2, 4, 8, 16
- **Sequence length:** 32, 128, 512, 1024, 2048 — capped at Stage 1's `context_length=256`
  until Stage 2 (a larger model) makes the longer end of this range meaningful.
- **Precision:** FP32, FP16, INT8, INT4 (as each becomes available — Phase 5)
- **Backend:** CPU, CUDA (as each becomes available — Phase 1, Phase 3)

## Recorded per run (reproducibility — Engineering Principle 5)

commit hash, model (name + config hash), GPU (or "CPU" + CPU model), CUDA version (if
applicable), compiler + version, precision, batch size, sequence length, runtime
configuration (execution mode from `docs/architecture.md` §8).

## Research questions this data should answer (spec §29)

1. How much speedup does KV caching provide? (Phase 2 ablation)
2. At what sequence length does GPU inference beat CPU?
3. When does quantization become memory-bandwidth-limited rather than compute-limited?
4. How does batch size affect throughput, and when does batching hurt latency?
5. What's the relationship between KV-cache size and sequence length?
6. Which ops (attention / MLP / norm / embedding) dominate execution time, at each scale?
7. How much does kernel fusion reduce memory traffic? (Phase 4 ablation)
8. Can the adaptive runtime beat every fixed policy it's composed from? (Phase 7)

## Status

No runtime exists yet to benchmark (Phase 0). `tools/benchmark` and
`benchmarks/results/` are placeholders; see `runtime/*/README.md` files for what
each phase adds.
