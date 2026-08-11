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
void transformer_block_forward(Tensor& x, const BlockWeights& w, int n_heads);

}  // namespace rt::transformer
