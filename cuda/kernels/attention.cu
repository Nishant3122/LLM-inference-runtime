#include "attention.cuh"

#include "../memory/device_buffer.h"
#include "../utils/cuda_check.h"
#include "linear.cuh"
#include "softmax.cuh"

namespace rt::cuda::transformer {

namespace {

// scores is [H*T, T] (row h*T+t1, col t2) so ops::softmax_rows can be reused on it
// unmodified — one thread per (h, t1, t2) triple.
__global__ void attention_scores_kernel(const float* q, const float* k, float* scores, int T,
                                         int D, int H, int hd, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * T;
    if (idx >= total) return;

    int h = idx / (T * T);
    int rem = idx % (T * T);
    int t1 = rem / T;
    int t2 = rem % T;

    float* out = scores + (static_cast<size_t>(h) * T + t1) * T + t2;
    if (t2 > t1) {
        *out = -INFINITY;
        return;
    }
    const float* q_row = q + static_cast<size_t>(t1) * D + h * hd;
    const float* k_row = k + static_cast<size_t>(t2) * D + h * hd;
    float dot = 0.0f;
    for (int i = 0; i < hd; ++i) dot += q_row[i] * k_row[i];
    *out = dot * scale;
}

// attn_out is [T, D] with heads as contiguous hd-chunks (same layout as CPU's
// attn_out buffer) — one thread per (t1, h, i) triple, i.e. per output element.
__global__ void attention_weighted_sum_kernel(const float* scores, const float* v, float* out,
                                               int T, int D, int H, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * D;
    if (idx >= total) return;

    int t1 = idx / D;
    int rem = idx % D;
    int h = rem / hd;
    int i = rem % hd;

    const float* weights_row = scores + (static_cast<size_t>(h) * T + t1) * T;
    float acc = 0.0f;
    for (int t2 = 0; t2 <= t1; ++t2) {
        const float* v_row = v + static_cast<size_t>(t2) * D + h * hd;
        acc += weights_row[t2] * v_row[i];
    }
    out[idx] = acc;
}

}  // namespace

void causal_self_attention(const rt::Tensor& x, const rt::transformer::AttentionWeights& w,
                            int n_heads, rt::Tensor& y) {
    const int T = static_cast<int>(x.shape[0]);
    const int D = static_cast<int>(x.shape[1]);
    const int H = n_heads;
    const int hd = D / H;
    const float scale = 1.0f / sqrtf(static_cast<float>(hd));

    rt::cuda::DeviceBuffer q_buf(static_cast<size_t>(T) * D * sizeof(float));
    rt::cuda::DeviceBuffer k_buf(static_cast<size_t>(T) * D * sizeof(float));
    rt::cuda::DeviceBuffer v_buf(static_cast<size_t>(T) * D * sizeof(float));
    rt::Tensor q(q_buf.data(), rt::Shape{static_cast<uint32_t>(T), static_cast<uint32_t>(D)},
                 rt::DataType::FP32, rt::Device::CUDA);
    rt::Tensor k(k_buf.data(), rt::Shape{static_cast<uint32_t>(T), static_cast<uint32_t>(D)},
                 rt::DataType::FP32, rt::Device::CUDA);
    rt::Tensor v(v_buf.data(), rt::Shape{static_cast<uint32_t>(T), static_cast<uint32_t>(D)},
                 rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::ops::linear(x, w.wq_w, w.wq_b, q);
    rt::cuda::ops::linear(x, w.wk_w, w.wk_b, k);
    rt::cuda::ops::linear(x, w.wv_w, w.wv_b, v);

    rt::cuda::DeviceBuffer scores_buf(static_cast<size_t>(H) * T * T * sizeof(float));
    {
        int total = H * T * T;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        attention_scores_kernel<<<blocks, threads>>>(
            static_cast<const float*>(q.data), static_cast<const float*>(k.data),
            static_cast<float*>(scores_buf.data()), T, D, H, hd, scale);
        CUDA_CHECK_LAST_ERROR();
    }

    rt::Tensor scores(scores_buf.data(),
                       rt::Shape{static_cast<uint32_t>(H * T), static_cast<uint32_t>(T)},
                       rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::ops::softmax_rows(scores);

    rt::cuda::DeviceBuffer attn_out_buf(static_cast<size_t>(T) * D * sizeof(float));
    {
        int total = T * D;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        attention_weighted_sum_kernel<<<blocks, threads>>>(
            static_cast<const float*>(scores_buf.data()), static_cast<const float*>(v.data),
            static_cast<float*>(attn_out_buf.data()), T, D, H, hd);
        CUDA_CHECK_LAST_ERROR();
    }

    rt::Tensor attn_out(attn_out_buf.data(),
                         rt::Shape{static_cast<uint32_t>(T), static_cast<uint32_t>(D)},
                         rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::ops::linear(attn_out, w.wo_w, w.wo_b, y);
}

}  // namespace rt::cuda::transformer
