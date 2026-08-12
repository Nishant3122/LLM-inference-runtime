// The transformer-block loop is inlined here rather than split into a separate
// cuda/kernels/transformer_block file: unlike attention/linear/softmax (each a
// genuine kernel), a "block" is just orchestration (norm -> attn -> residual, norm ->
// mlp -> residual) — same reasoning as cpu_backend.cpp keeping forward_impl in one
// function rather than scattering it.
#include "cuda_backend.h"

#include "../../cuda/kernels/attention.cuh"
#include "../../cuda/kernels/elementwise.cuh"
#include "../../cuda/kernels/embedding.cuh"
#include "../../cuda/kernels/layer_norm.cuh"
#include "../../cuda/kernels/linear.cuh"
#include "../../cuda/memory/device_buffer.h"

namespace rt::execution {

namespace {
rt::Tensor make_tensor(rt::cuda::DeviceBuffer& buf, uint32_t rows, uint32_t cols) {
    return rt::Tensor(buf.data(), rt::Shape{rows, cols}, rt::DataType::FP32, rt::Device::CUDA);
}
}  // namespace

ForwardResult forward_cuda(const cuda::CudaModel& model, const std::vector<int32_t>& ids) {
    ForwardResult result;
    result.T = static_cast<uint32_t>(ids.size());
    result.D = model.config.d_model;
    result.V = model.config.vocab_size;
    const size_t td_bytes = static_cast<size_t>(result.T) * result.D * sizeof(float);

    cuda::DeviceBuffer x_buf(td_bytes);
    rt::Tensor x = make_tensor(x_buf, result.T, result.D);
    cuda::transformer::embedding_forward(model.tok_embedding, model.pos_embedding, ids, x);

    result.embedding_output.resize(static_cast<size_t>(result.T) * result.D);
    x_buf.download(result.embedding_output.data(), td_bytes);

    cuda::DeviceBuffer normed_buf(td_bytes);
    cuda::DeviceBuffer sub_out_buf(td_bytes);
    rt::Tensor normed = make_tensor(normed_buf, result.T, result.D);
    rt::Tensor sub_out = make_tensor(sub_out_buf, result.T, result.D);

    result.block_outputs.resize(model.layers.size());
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];

        // x = x + Attention(LN1(x))
        cuda::transformer::layer_norm(x, layer.ln1_w, layer.ln1_b, normed);
        cuda::transformer::causal_self_attention(normed, layer.attn,
                                                   static_cast<int>(model.config.n_heads), sub_out);
        cuda::ops::add_inplace(x, sub_out);

        // x = x + MLP(LN2(x))
        cuda::transformer::layer_norm(x, layer.ln2_w, layer.ln2_b, normed);
        const uint32_t d_ff = layer.mlp.fc1_w.shape[1];
        cuda::DeviceBuffer hidden_buf(static_cast<size_t>(result.T) * d_ff * sizeof(float));
        rt::Tensor hidden = make_tensor(hidden_buf, result.T, d_ff);
        cuda::ops::linear(normed, layer.mlp.fc1_w, layer.mlp.fc1_b, hidden);
        cuda::ops::gelu_inplace(hidden);
        cuda::ops::linear(hidden, layer.mlp.fc2_w, layer.mlp.fc2_b, sub_out);
        cuda::ops::add_inplace(x, sub_out);

        result.block_outputs[i].resize(static_cast<size_t>(result.T) * result.D);
        x_buf.download(result.block_outputs[i].data(), td_bytes);
    }

    cuda::DeviceBuffer final_norm_buf(td_bytes);
    rt::Tensor final_norm = make_tensor(final_norm_buf, result.T, result.D);
    cuda::transformer::layer_norm(x, model.ln_f_w, model.ln_f_b, final_norm);
    result.final_norm_output.resize(static_cast<size_t>(result.T) * result.D);
    final_norm_buf.download(result.final_norm_output.data(), td_bytes);

    const size_t tv_bytes = static_cast<size_t>(result.T) * result.V * sizeof(float);
    cuda::DeviceBuffer logits_buf(tv_bytes);
    rt::Tensor logits = make_tensor(logits_buf, result.T, result.V);
    cuda::ops::linear(final_norm, model.lm_head_w, model.lm_head_b, logits);
    result.logits.resize(static_cast<size_t>(result.T) * result.V);
    logits_buf.download(result.logits.data(), tv_bytes);

    return result;
}

}  // namespace rt::execution
