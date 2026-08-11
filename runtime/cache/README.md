# runtime/cache — Phase 2 ✅

- `kv_cache.h/.cpp` — `LayerKVCache`: one preallocated `[context_length, d_model]`
  buffer per layer for K and V, `append()`/`reset()`, grown by exactly one row per
  decode step. `KVCache`: one `LayerKVCache` per transformer layer. Preallocated at
  `init()` to `context_length` rows so decode steps never `malloc`/reallocate (spec §13).
- `cache_manager.h/.cpp` — `create_cache()` + `estimate_kv_cache_bytes()`. Deliberately
  thin: admission control, multi-sequence pooling, and eviction (spec §12) only earn
  their complexity once Phase 6 (batching) has multiple concurrent sequences to manage
  — see `runtime/scheduler/README.md`. No sliding-window eviction yet either:
  `LayerKVCache::append()` throws once `context_length` is exceeded rather than
  silently dropping old entries; `examples/generate.cpp` stops generation gracefully
  when it hits that limit.

## Correctness: cache vs. recompute-from-scratch

`tests/kv_cache_test.cpp` feeds the same token sequence through both
`execution::forward()` (Phase 1, recomputes everything) and `execution::prefill()` +
`execution::decode_step()` (Phase 2, cached) and diffs every step's logits. **Result:
`max_abs_diff = 0` at every single step** — not just close, exactly bit-identical.
This is expected, not lucky: every op here (LayerNorm, linear, attention's weighted
sum) computes a given row independently of any other row being computed alongside it,
and the cached path's summation order for a row is identical to the uncached path's
(masked-out future positions contribute exact `+0.0` to the uncached path's sums, so
excluding them entirely, as the cached path does, changes nothing) — see the comment
on `causal_self_attention_cached` in `runtime/transformer/attention.h`.

## Performance: with cache vs. without (spec §32 ablation, Q1)

Measured via `examples/generate.cpp` (prompt "The cat", greedy-ish sampling,
temperature 0.8, top-k 20) on this machine:

| Tokens generated | Without cache (Phase 1) | With cache (Phase 2) | Speedup |
|---|---|---|---|
| 50  | 2.114s (23.7 tok/s) | 0.086s (582.0 tok/s) | **24.6x** |
| 200 | 23.208s (8.6 tok/s) | 0.321s (622.4 tok/s) | **72.3x** |

Speedup grows with sequence length, as expected: the uncached path redoes the linear
projections (`wq`/`wk`/`wv`/`wo`/`fc1`/`fc2` — the dominant cost, since `d_ff=512` is
4x `d_model`) for *every* position on *every* step, an `O(N^2)`-total cost; the cached
path does `O(1)` new linear-layer work per step (one new row) and only attention
itself still grows with the cache length. This is exactly answers spec §29's Q1 ("how
much speedup does KV caching provide?") for the Stage-1 model — expect the gap to
widen further on a larger model where `d_model`/`d_ff` dominate even more relative to
attention's `O(L)`-per-step cost.
