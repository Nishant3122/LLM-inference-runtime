#include "layer_norm.cuh"

#include "../utils/cuda_check.h"

namespace rt::cuda::transformer {

namespace {

__global__ void layer_norm_kernel(const float* x, const float* weight, const float* bias,
                                   float* y, int T, int D, float eps) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const float* row = x + static_cast<size_t>(t) * D;
    float* out_row = y + static_cast<size_t>(t) * D;

    double mean = 0.0;
    for (int d = 0; d < D; ++d) mean += row[d];
    mean /= D;

    double var = 0.0;
    for (int d = 0; d < D; ++d) {
        double diff = row[d] - mean;
        var += diff * diff;
    }
    var /= D;

    float inv_std = static_cast<float>(1.0 / sqrt(var + eps));
    for (int d = 0; d < D; ++d) {
        float normalized = static_cast<float>(row[d] - mean) * inv_std;
        out_row[d] = normalized * weight[d] + bias[d];
    }
}

}  // namespace

void layer_norm(const rt::Tensor& x, const rt::Tensor& weight, const rt::Tensor& bias,
                 rt::Tensor& y, float eps) {
    const int T = static_cast<int>(x.shape[0]);
    const int D = static_cast<int>(x.shape[1]);

    const int threads = 256;
    const int blocks = (T + threads - 1) / threads;
    layer_norm_kernel<<<blocks, threads>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(weight.data),
        static_cast<const float*>(bias.data), static_cast<float*>(y.data), T, D, eps);
    CUDA_CHECK_LAST_ERROR();
}

}  // namespace rt::cuda::transformer
