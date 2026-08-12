#include "softmax.cuh"

#include <cfloat>

#include "../utils/cuda_check.h"

namespace rt::cuda::ops {

namespace {

__global__ void softmax_rows_kernel(float* x, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float* row = x + static_cast<size_t>(r) * cols;

    float max_val = -FLT_MAX;
    for (int c = 0; c < cols; ++c) max_val = fmaxf(max_val, row[c]);

    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = expf(row[c] - max_val);
        row[c] = e;
        sum += e;
    }
    if (sum > 0.0f) {
        for (int c = 0; c < cols; ++c) row[c] /= sum;
    }
}

}  // namespace

void softmax_rows(rt::Tensor& x) {
    const int rows = static_cast<int>(x.shape[0]);
    const int cols = static_cast<int>(x.shape[1]);

    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;
    softmax_rows_kernel<<<blocks, threads>>>(static_cast<float*>(x.data), rows, cols);
    CUDA_CHECK_LAST_ERROR();
}

}  // namespace rt::cuda::ops
