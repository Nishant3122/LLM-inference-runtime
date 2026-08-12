#include "cuda_model.h"

namespace rt::cuda {

namespace {

// host_t.data points somewhere inside host_model.storage's contiguous buffer (that's
// how model_loader.cpp built every named Tensor); re-point the same shape/dtype at
// the equivalent offset in device memory.
rt::Tensor remap(const rt::Tensor& host_t, const uint8_t* host_base, DeviceBuffer& device_storage) {
    size_t offset = static_cast<const uint8_t*>(host_t.data) - host_base;
    rt::Tensor dev_t;
    dev_t.data = device_storage.at<void>(offset);
    dev_t.shape = host_t.shape;
    dev_t.dtype = host_t.dtype;
    dev_t.device = rt::Device::CUDA;
    dev_t.layout = host_t.layout;
    return dev_t;
}

}  // namespace

CudaModel upload_to_cuda(const model::Model& host_model) {
    CudaModel gpu;
    gpu.config = host_model.config;

    gpu.storage = DeviceBuffer(host_model.storage.size());
    gpu.storage.upload(host_model.storage.data(), host_model.storage.size());
    const uint8_t* host_base = host_model.storage.data();

    gpu.tok_embedding = remap(host_model.tok_embedding, host_base, gpu.storage);
    gpu.pos_embedding = remap(host_model.pos_embedding, host_base, gpu.storage);

    gpu.layers.resize(host_model.layers.size());
    for (size_t i = 0; i < host_model.layers.size(); ++i) {
        const auto& hl = host_model.layers[i];
        auto& gl = gpu.layers[i];
        gl.ln1_w = remap(hl.ln1_w, host_base, gpu.storage);
        gl.ln1_b = remap(hl.ln1_b, host_base, gpu.storage);
        gl.attn.wq_w = remap(hl.attn.wq_w, host_base, gpu.storage);
        gl.attn.wq_b = remap(hl.attn.wq_b, host_base, gpu.storage);
        gl.attn.wk_w = remap(hl.attn.wk_w, host_base, gpu.storage);
        gl.attn.wk_b = remap(hl.attn.wk_b, host_base, gpu.storage);
        gl.attn.wv_w = remap(hl.attn.wv_w, host_base, gpu.storage);
        gl.attn.wv_b = remap(hl.attn.wv_b, host_base, gpu.storage);
        gl.attn.wo_w = remap(hl.attn.wo_w, host_base, gpu.storage);
        gl.attn.wo_b = remap(hl.attn.wo_b, host_base, gpu.storage);
        gl.ln2_w = remap(hl.ln2_w, host_base, gpu.storage);
        gl.ln2_b = remap(hl.ln2_b, host_base, gpu.storage);
        gl.mlp.fc1_w = remap(hl.mlp.fc1_w, host_base, gpu.storage);
        gl.mlp.fc1_b = remap(hl.mlp.fc1_b, host_base, gpu.storage);
        gl.mlp.fc2_w = remap(hl.mlp.fc2_w, host_base, gpu.storage);
        gl.mlp.fc2_b = remap(hl.mlp.fc2_b, host_base, gpu.storage);
    }

    gpu.ln_f_w = remap(host_model.ln_f_w, host_base, gpu.storage);
    gpu.ln_f_b = remap(host_model.ln_f_b, host_base, gpu.storage);
    gpu.lm_head_w = remap(host_model.lm_head_w, host_base, gpu.storage);
    gpu.lm_head_b = remap(host_model.lm_head_b, host_base, gpu.storage);

    return gpu;
}

}  // namespace rt::cuda
