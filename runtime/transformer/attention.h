// Naive causal multi-head self-attention, matching
// reference/model.py::CausalSelfAttention exactly:
//   Q = XWq, K = XWk, V = XWv
//   scores = QK^T / sqrt(head_dim), causal-masked
//   weights = softmax(scores)
//   out = Wo(weights @ V)
//
// "Naive" per spec §11: O(T^2) scores matrix materialized in full, no fused kernel,
// no KV cache (that's Phase 2). Optimize only after this is validated against
// tests/golden/*/block_*_output.npy.
#pragma once

#include "../cache/kv_cache.h"
#include "../core/tensor.h"

namespace rt::transformer {

struct AttentionWeights {
    Tensor wq_w, wq_b;  // wq_w: [D, D], wq_b: [D]
    Tensor wk_w, wk_b;
    Tensor wv_w, wv_b;
    Tensor wo_w, wo_b;
};

// x: [T, D]: y: [T, D] (pre-allocated, may alias nothing — caller owns storage).
// n_heads must divide D (checked at Model load time, re-asserted here defensively).
//
// If kv_out != nullptr, this call's computed K and V (for all T positions) are also
// appended to it — used by Phase 2's prefill to seed the cache while computing the
// prompt's forward pass exactly as Phase 1 always has (same operations, same order,
// so results are unchanged and tests/model_test.cpp keeps passing untouched).
void causal_self_attention(const Tensor& x, const AttentionWeights& w, int n_heads, Tensor& y,
                            cache::LayerKVCache* kv_out = nullptr);

// Phase 2 decode step: x is exactly one token's pre-attention activations ([1, D]).
// Computes this token's Q/K/V, appends K/V to `kv` (kv.length() grows by 1), then
// attends Q against kv's full history — all positions [0, kv.length()) after the
// append, which includes this token itself. Numerically this produces exactly the
// row causal_self_attention() would produce for this token if the whole sequence
// were recomputed from scratch (see runtime/cache/README.md for why: per-row ops
// don't depend on what other rows are being computed alongside them).
void causal_self_attention_cached(const Tensor& x, const AttentionWeights& w, int n_heads,
                                   cache::LayerKVCache& kv, Tensor& y);

}  // namespace rt::transformer
