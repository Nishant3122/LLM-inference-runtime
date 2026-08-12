#include "linear.cuh"

#include "../utils/cuda_check.h"

namespace rt::cuda::ops {

namespace {

__global__ void linear_kernel(const float* x, const float* w, const float* b, float* y,
                               int rows, int in_features, int out_features, bool has_bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * out_features;
    if (idx >= total) return;

    int r = idx / out_features;
    int o = idx % out_features;
    const float* xrow = x + static_cast<size_t>(r) * in_features;

    float acc = has_bias ? b[o] : 0.0f;
    for (int i = 0; i < in_features; ++i) {
        acc += xrow[i] * w[static_cast<size_t>(i) * out_features + o];
    }
    y[idx] = acc;
}

// Duplicated from elementwise.cu's gelu_device rather than shared via a header:
// it's a one-line formula, and each naive kernel file being self-contained matches
// how the rest of cuda/kernels is written (see e.g. attention.cu's own dot-product
// loop instead of calling into linear's kernel internals).
__device__ __forceinline__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));  // 1/sqrt(2)
}

__global__ void linear_fused_kernel(const float* x, const float* w, const float* b,
                                     const float* residual, float* y, int rows, int in_features,
                                     int out_features, bool has_bias, bool apply_gelu,
                                     bool has_residual) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * out_features;
    if (idx >= total) return;

    int r = idx / out_features;
    int o = idx % out_features;
    const float* xrow = x + static_cast<size_t>(r) * in_features;

    float acc = has_bias ? b[o] : 0.0f;
    for (int i = 0; i < in_features; ++i) {
        acc += xrow[i] * w[static_cast<size_t>(i) * out_features + o];
    }
    if (apply_gelu) acc = gelu_device(acc);
    if (has_residual) acc += residual[idx];
    y[idx] = acc;
}

}  // namespace

void linear(const rt::Tensor& x, const rt::Tensor& W, const rt::Tensor& b, rt::Tensor& y) {
    const int rows = static_cast<int>(x.shape[0]);
    const int in_features = static_cast<int>(x.shape[1]);
    const int out_features = static_cast<int>(W.shape[1]);
    const int total = rows * out_features;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(W.data),
        b.data != nullptr ? static_cast<const float*>(b.data) : nullptr,
        static_cast<float*>(y.data), rows, in_features, out_features, b.data != nullptr);
    CUDA_CHECK_LAST_ERROR();
}

void linear_fused(const rt::Tensor& x, const rt::Tensor& W, const rt::Tensor& b, rt::Tensor& y,
                   bool apply_gelu, const rt::Tensor* residual) {
    const int rows = static_cast<int>(x.shape[0]);
    const int in_features = static_cast<int>(x.shape[1]);
    const int out_features = static_cast<int>(W.shape[1]);
    const int total = rows * out_features;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    linear_fused_kernel<<<blocks, threads>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(W.data),
        b.data != nullptr ? static_cast<const float*>(b.data) : nullptr,
        residual != nullptr ? static_cast<const float*>(residual->data) : nullptr,
        static_cast<float*>(y.data), rows, in_features, out_features, b.data != nullptr,
        apply_gelu, residual != nullptr);
    CUDA_CHECK_LAST_ERROR();
}

}  // namespace rt::cuda::ops
