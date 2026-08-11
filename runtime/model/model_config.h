// Mirrors reference/model.py::TinyTransformerConfig and the model.bin header fields
// (docs/model_format.md). Kept as plain data — no behavior — so it's trivial to
// construct both from a loaded file (model_loader) and, in tests, by hand.
#pragma once

#include <cstdint>
#include <stdexcept>

namespace rt::model {

enum class Architecture : uint32_t {
    TinyTransformerDecoder = 0,
};

struct ModelConfig {
    Architecture architecture = Architecture::TinyTransformerDecoder;
    uint32_t vocab_size = 0;
    uint32_t d_model = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t d_ff = 0;
    uint32_t context_length = 0;

    uint32_t head_dim() const {
        if (n_heads == 0 || d_model % n_heads != 0) {
            throw std::runtime_error("ModelConfig: d_model must be divisible by n_heads");
        }
        return d_model / n_heads;
    }
};

}  // namespace rt::model
