#include "cpu_backend.h"

#include "../transformer/embedding.h"
#include "../transformer/normalization.h"
#include "../transformer/ops.h"
#include "../transformer/transformer_block.h"

namespace rt::execution {

ForwardResult forward(const model::Model& model, const std::vector<int32_t>& ids) {
    ForwardResult result;
    result.T = static_cast<uint32_t>(ids.size());
    result.D = model.config.d_model;
    result.V = model.config.vocab_size;

    result.embedding_output.resize(static_cast<size_t>(result.T) * result.D);
    Tensor emb_t(result.embedding_output.data(), Shape{result.T, result.D}, DataType::FP32,
                 Device::CPU);
    transformer::embedding_forward(model.tok_embedding, model.pos_embedding, ids, emb_t);

    // x is mutated in place through each block (residual adds happen on this buffer);
    // block_outputs[i] is a snapshot copy for golden-output comparison.
    std::vector<float> x = result.embedding_output;
    Tensor x_t(x.data(), Shape{result.T, result.D}, DataType::FP32, Device::CPU);

    result.block_outputs.resize(model.layers.size());
    for (size_t i = 0; i < model.layers.size(); ++i) {
        transformer::transformer_block_forward(x_t, model.layers[i],
                                                 static_cast<int>(model.config.n_heads));
        result.block_outputs[i] = x;
    }

    result.final_norm_output.resize(static_cast<size_t>(result.T) * result.D);
    Tensor final_norm_t(result.final_norm_output.data(), Shape{result.T, result.D}, DataType::FP32,
                         Device::CPU);
    transformer::layer_norm(x_t, model.ln_f_w, model.ln_f_b, final_norm_t);

    result.logits.resize(static_cast<size_t>(result.T) * result.V);
    Tensor logits_t(result.logits.data(), Shape{result.T, result.V}, DataType::FP32, Device::CPU);
    ops::linear(final_norm_t, model.lm_head_w, model.lm_head_b, logits_t);

    return result;
}

}  // namespace rt::execution
