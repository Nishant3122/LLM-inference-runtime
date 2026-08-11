// The loaded, in-memory model: owns the raw weight bytes (`storage_`) and exposes
// named Tensor views into it (embeddings, per-layer BlockWeights, final norm, LM
// head). Combines what the original spec's model_loader/weight_manager split would
// have been into one type for Phase 1 — see runtime/model/README.md for why: with
// only one owner and one load path, a separate WeightManager class would just be
// Model with extra indirection (Engineering Principle 4, avoid premature abstraction).
// Revisit the split if/when the CUDA backend needs its own device-side ownership
// story that doesn't fit this shape.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/tensor.h"
#include "../transformer/transformer_block.h"
#include "model_config.h"

namespace rt::model {

class Model {
public:
    Model() = default;

    // Owns storage_; views into it are invalidated by copying (storage_ would be
    // reallocated). Moves are fine (vector's heap buffer address is preserved).
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    ModelConfig config;

    Tensor tok_embedding;  // [vocab_size, d_model]
    Tensor pos_embedding;  // [context_length, d_model]
    std::vector<transformer::BlockWeights> layers;
    Tensor ln_f_w, ln_f_b;   // [d_model]
    Tensor lm_head_w, lm_head_b;  // lm_head_w: [d_model, vocab_size], lm_head_b: [vocab_size]

    // Raw byte storage every Tensor above points into. Public so model_loader can
    // populate it; not meant to be touched after loading.
    std::vector<uint8_t> storage;

    // Debug/introspection: every tensor by its model.bin name (tools that want to
    // dump/inspect arbitrary tensors don't need new Model fields for that).
    std::unordered_map<std::string, Tensor> tensors_by_name;
};

}  // namespace rt::model
