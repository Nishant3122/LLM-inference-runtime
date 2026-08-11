// One pre-norm Transformer block, matching reference/model.py::TransformerBlock:
//   x = x + Attention(LayerNorm1(x))
//   x = x + MLP(LayerNorm2(x))
#pragma once

#include "../core/tensor.h"
#include "attention.h"
#include "mlp.h"

namespace rt::transformer {

struct BlockWeights {
    Tensor ln1_w, ln1_b;
    AttentionWeights attn;
    Tensor ln2_w, ln2_b;
    MlpWeights mlp;
};

// x: [T, D], updated in place (residual adds applied directly to x's buffer).
// If kv_out != nullptr, this layer's K/V for all T positions are appended to it
// (Phase 2 prefill — see causal_self_attention's kv_out parameter).
void transformer_block_forward(Tensor& x, const BlockWeights& w, int n_heads,
                                cache::LayerKVCache* kv_out = nullptr);

// Phase 2 decode step: x is exactly one token's activations ([1, D]), updated in
// place. Uses (and grows) this layer's persistent KV cache instead of recomputing
// attention over the whole sequence.
void transformer_block_forward_cached(Tensor& x, const BlockWeights& w, int n_heads,
                                       cache::LayerKVCache& kv);

}  // namespace rt::transformer
