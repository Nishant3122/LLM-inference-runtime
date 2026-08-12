#include "embedding.cuh"

#include "../memory/device_buffer.h"
#include "../utils/cuda_check.h"

namespace rt::cuda::transformer {

namespace {

__global__ void embedding_kernel(const float* tok, const float* pos, const int32_t* ids,
                                  float* y, int T, int D, uint32_t position_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * D;
    if (idx >= total) return;

    int t = idx / D;
    int d = idx % D;
    int id = ids[t];
    y[idx] = tok[static_cast<size_t>(id) * D + d] +
             pos[static_cast<size_t>(position_offset + t) * D + d];
}

}  // namespace

void embedding_forward(const rt::Tensor& tok_embedding, const rt::Tensor& pos_embedding,
                        const std::vector<int32_t>& ids, rt::Tensor& y,
                        uint32_t position_offset) {
    const int T = static_cast<int>(ids.size());
    const int D = static_cast<int>(tok_embedding.shape[1]);

    rt::cuda::DeviceBuffer ids_dev(ids.size() * sizeof(int32_t));
    ids_dev.upload(ids.data(), ids.size() * sizeof(int32_t));

    const int total = T * D;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(
        static_cast<const float*>(tok_embedding.data), static_cast<const float*>(pos_embedding.data),
        ids_dev.at<const int32_t>(0), static_cast<float*>(y.data), T, D, position_offset);
    CUDA_CHECK_LAST_ERROR();
}

}  // namespace rt::cuda::transformer
