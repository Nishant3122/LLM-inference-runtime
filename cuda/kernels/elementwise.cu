#include "elementwise.cuh"

#include "../utils/cuda_check.h"

namespace rt::cuda::ops {

namespace {

__device__ __forceinline__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));  // 1/sqrt(2)
}

__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu_device(x[i]);
}

__global__ void add_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

}  // namespace

void gelu_inplace(rt::Tensor& x) {
    const int n = static_cast<int>(x.numel());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(static_cast<float*>(x.data), n);
    CUDA_CHECK_LAST_ERROR();
}

void add_inplace(rt::Tensor& x, const rt::Tensor& y) {
    const int n = static_cast<int>(x.numel());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(static_cast<float*>(x.data), static_cast<const float*>(y.data),
                                     n);
    CUDA_CHECK_LAST_ERROR();
}

}  // namespace rt::cuda::ops
