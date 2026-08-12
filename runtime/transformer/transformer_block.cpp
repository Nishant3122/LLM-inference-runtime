#include "transformer_block.h"

#include <cassert>
#include <vector>

#include "normalization.h"

namespace rt::transformer {

void transformer_block_forward(Tensor& x, const BlockWeights& w, int n_heads,
                                cache::LayerKVCache* kv_out, profiling::Profiler* profiler) {
    assert(x.shape.ndim == 2);
    const uint32_t T = x.shape[0];
    const uint32_t D = x.shape[1];
    float* xd = static_cast<float*>(x.data);

    std::vector<float> normed(static_cast<size_t>(T) * D);
    std::vector<float> sub_out(static_cast<size_t>(T) * D);
    Tensor normed_t(normed.data(), Shape{T, D}, DataType::FP32, Device::CPU);
    Tensor sub_out_t(sub_out.data(), Shape{T, D}, DataType::FP32, Device::CPU);

    // x = x + Attention(LayerNorm1(x))
    profiling::time_if(profiler, "layer_norm",
                        [&] { layer_norm(x, w.ln1_w, w.ln1_b, normed_t); });
    profiling::time_if(profiler, "attention", [&] {
        causal_self_attention(normed_t, w.attn, n_heads, sub_out_t, kv_out);
    });
    profiling::time_if(profiler, "residual_add", [&] {
        for (size_t i = 0; i < normed.size(); ++i) xd[i] += sub_out[i];
    });

    // x = x + MLP(LayerNorm2(x))
    profiling::time_if(profiler, "layer_norm",
                        [&] { layer_norm(x, w.ln2_w, w.ln2_b, normed_t); });
    profiling::time_if(profiler, "mlp", [&] { mlp_forward(normed_t, w.mlp, sub_out_t); });
    profiling::time_if(profiler, "residual_add", [&] {
        for (size_t i = 0; i < normed.size(); ++i) xd[i] += sub_out[i];
    });
}

void transformer_block_forward_cached(Tensor& x, const BlockWeights& w, int n_heads,
                                       cache::LayerKVCache& kv) {
    assert(x.shape.ndim == 2 && x.shape[0] == 1);
    const uint32_t D = x.shape[1];
    float* xd = static_cast<float*>(x.data);

    std::vector<float> normed(D);
    std::vector<float> sub_out(D);
    Tensor normed_t(normed.data(), Shape{1, D}, DataType::FP32, Device::CPU);
    Tensor sub_out_t(sub_out.data(), Shape{1, D}, DataType::FP32, Device::CPU);

    // x = x + Attention(LayerNorm1(x))  -- cached: attends over kv's full history
    layer_norm(x, w.ln1_w, w.ln1_b, normed_t);
    causal_self_attention_cached(normed_t, w.attn, n_heads, kv, sub_out_t);
    for (size_t i = 0; i < normed.size(); ++i) xd[i] += sub_out[i];

    // x = x + MLP(LayerNorm2(x))  -- unchanged: MLP has no cross-token dependency
    layer_norm(x, w.ln2_w, w.ln2_b, normed_t);
    mlp_forward(normed_t, w.mlp, sub_out_t);
    for (size_t i = 0; i < normed.size(); ++i) xd[i] += sub_out[i];
}

}  // namespace rt::transformer
