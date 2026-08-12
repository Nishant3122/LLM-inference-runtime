// Device-resident mirror of runtime/model/model.h::Model. Same shape (config +
// named Tensor views for every weight), but storage_ is device memory instead of a
// host std::vector — reuses rt::transformer::BlockWeights/AttentionWeights/MlpWeights
// unchanged since Tensor already carries a Device tag (docs/architecture.md §6).
#pragma once

#include <vector>

#include "../../runtime/model/model.h"
#include "../../runtime/model/model_config.h"
#include "../../runtime/transformer/transformer_block.h"
#include "device_buffer.h"

namespace rt::cuda {

struct CudaModel {
    CudaModel() = default;
    CudaModel(const CudaModel&) = delete;
    CudaModel& operator=(const CudaModel&) = delete;
    CudaModel(CudaModel&&) = default;
    CudaModel& operator=(CudaModel&&) = default;

    model::ModelConfig config;

    rt::Tensor tok_embedding;
    rt::Tensor pos_embedding;
    std::vector<rt::transformer::BlockWeights> layers;
    rt::Tensor ln_f_w, ln_f_b;
    rt::Tensor lm_head_w, lm_head_b;

    DeviceBuffer storage;  // owns every Tensor's data above
};

// One cudaMalloc + one cudaMemcpy for the whole weight blob (mirrors how
// model_loader reads model.bin into one contiguous host buffer, see
// docs/model_format.md), then rebuilds the same named-tensor views with device
// pointers instead of host ones. Called once per model load, not per token.
CudaModel upload_to_cuda(const model::Model& host_model);

}  // namespace rt::cuda
