#include "transformer_block.h"

#include <cassert>
#include <vector>

#include "normalization.h"

namespace rt::transformer {

void transformer_block_forward(Tensor& x, const BlockWeights& w, int n_heads) {
    assert(x.shape.ndim == 2);
    const uint32_t T = x.shape[0];
    const uint32_t D = x.shape[1];
    float* xd = static_cast<float*>(x.data);

    std::vector<float> normed(static_cast<size_t>(T) * D);
    std::vector<float> sub_out(static_cast<size_t>(T) * D);
    Tensor normed_t(normed.data(), Shape{T, D}, DataType::FP32, Device::CPU);
    Tensor sub_out_t(sub_out.data(), Shape{T, D}, DataType::FP32, Device::CPU);

    // x = x + Attention(LayerNorm1(x))
    layer_norm(x, w.ln1_w, w.ln1_b, normed_t);
    causal_self_attention(normed_t, w.attn, n_heads, sub_out_t);
    for (size_t i = 0; i < normed.size(); ++i) xd[i] += sub_out[i];

    // x = x + MLP(LayerNorm2(x))
    layer_norm(x, w.ln2_w, w.ln2_b, normed_t);
    mlp_forward(normed_t, w.mlp, sub_out_t);
    for (size_t i = 0; i < normed.size(); ++i) xd[i] += sub_out[i];
}

}  // namespace rt::transformer
