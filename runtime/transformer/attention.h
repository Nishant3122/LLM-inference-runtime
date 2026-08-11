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
void causal_self_attention(const Tensor& x, const AttentionWeights& w, int n_heads, Tensor& y);

}  // namespace rt::transformer
