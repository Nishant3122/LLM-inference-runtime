# runtime/cache — Phase 2

Not yet implemented. Will contain:

- `kv_cache` — per-layer K/V buffers for autoregressive generation (`docs/architecture.md`
  §5 config: 4 layers, `d_model=128`, 4 heads -> per-token per-layer K/V is small enough
  that Stage 1's cache management story can be validated without a memory pool first,
  before that becomes necessary for larger models).
- `cache_manager` — allocation, growth, indexing, memory reuse, sequence termination,
  batch requests, eviction (spec §12).

First correctness target: generation with the cache produces byte-identical logits to
generation that recomputes K/V from scratch every step (the "without caching" path in
spec §12) — that's the ablation in spec §32 (KV Cache ON/OFF). The "without caching"
side of that ablation already exists: Phase 1's `runtime/execution/cpu_backend`
recomputes the full sequence every step (see its README's ad hoc timing, ~2.4s for 50
tokens on this machine) — that's the baseline number Phase 2 needs to beat.
