#include "attention.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ops.h"

namespace rt::transformer {

void causal_self_attention(const Tensor& x, const AttentionWeights& w, int n_heads, Tensor& y) {
    assert(x.shape.ndim == 2);
    const uint32_t T = x.shape[0];
    const uint32_t D = x.shape[1];
    if (n_heads <= 0 || D % static_cast<uint32_t>(n_heads) != 0) {
        throw std::runtime_error("causal_self_attention: d_model must be divisible by n_heads");
    }
    const uint32_t H = static_cast<uint32_t>(n_heads);
    const uint32_t hd = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Q = XWq, K = XWk, V = XWv  -- each [T, D], heads are contiguous hd-chunks of D
    // (matches reference/model.py's `.view(B, T, H, hd)` on the last dimension).
    std::vector<float> q_buf(static_cast<size_t>(T) * D);
    std::vector<float> k_buf(static_cast<size_t>(T) * D);
    std::vector<float> v_buf(static_cast<size_t>(T) * D);
    Tensor q(q_buf.data(), Shape{T, D}, DataType::FP32, Device::CPU);
    Tensor k(k_buf.data(), Shape{T, D}, DataType::FP32, Device::CPU);
    Tensor v(v_buf.data(), Shape{T, D}, DataType::FP32, Device::CPU);
    ops::linear(x, w.wq_w, w.wq_b, q);
    ops::linear(x, w.wk_w, w.wk_b, k);
    ops::linear(x, w.wv_w, w.wv_b, v);

    // attn_out[t, h*hd + i] accumulates head h's output before the wo projection.
    std::vector<float> attn_out(static_cast<size_t>(T) * D, 0.0f);
    std::vector<float> scores(static_cast<size_t>(T) * T);  // reused per head

    for (uint32_t h = 0; h < H; ++h) {
        const size_t head_off = static_cast<size_t>(h) * hd;

        // scores[t1, t2] = (q_t1 . k_t2) * scale, causal-masked (t2 > t1 -> -inf)
        for (uint32_t t1 = 0; t1 < T; ++t1) {
            const float* q_row = q_buf.data() + static_cast<size_t>(t1) * D + head_off;
            for (uint32_t t2 = 0; t2 < T; ++t2) {
                if (t2 > t1) {
                    scores[static_cast<size_t>(t1) * T + t2] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float* k_row = k_buf.data() + static_cast<size_t>(t2) * D + head_off;
                float dot = 0.0f;
                for (uint32_t i = 0; i < hd; ++i) dot += q_row[i] * k_row[i];
                scores[static_cast<size_t>(t1) * T + t2] = dot * scale;
            }
        }

        Tensor scores_t(scores.data(), Shape{T, T}, DataType::FP32, Device::CPU);
        ops::softmax_rows(scores_t);

        // out_h[t1, i] = sum_t2 weights[t1, t2] * v[t2, i]
        for (uint32_t t1 = 0; t1 < T; ++t1) {
            const float* weights_row = scores.data() + static_cast<size_t>(t1) * T;
            float* out_row = attn_out.data() + static_cast<size_t>(t1) * D + head_off;
            for (uint32_t t2 = 0; t2 <= t1; ++t2) {  // weights are 0 for t2 > t1
                float wgt = weights_row[t2];
                if (wgt == 0.0f) continue;
                const float* v_row = v_buf.data() + static_cast<size_t>(t2) * D + head_off;
                for (uint32_t i = 0; i < hd; ++i) out_row[i] += wgt * v_row[i];
            }
        }
    }

    Tensor attn_out_t(attn_out.data(), Shape{T, D}, DataType::FP32, Device::CPU);
    ops::linear(attn_out_t, w.wo_w, w.wo_b, y);
}

}  // namespace rt::transformer
