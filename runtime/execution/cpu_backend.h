// Drives the full forward pass: embedding -> N transformer blocks -> final norm ->
// LM head, mirroring reference/model.py::TinyTransformer.forward exactly.
//
// Always returns every intermediate (not just final logits) even though
// examples/generate.cpp only needs the last row of `logits`: Phase 1's whole point is
// bisectable correctness against tests/golden/ (see reference/dump_golden.py), and at
// this scale (a few hundred KB of activations) the extra copies are not worth
// optimizing away before Phase 4 profiling says they matter (Engineering Principle 1).
#pragma once

#include <vector>

#include "../model/model.h"

namespace rt::execution {

struct ForwardResult {
    uint32_t T = 0, D = 0, V = 0;
    std::vector<float> embedding_output;             // [T, D]
    std::vector<std::vector<float>> block_outputs;    // n_layers x [T, D]
    std::vector<float> final_norm_output;             // [T, D]
    std::vector<float> logits;                        // [T, V]
};

// ids.size() must not exceed model.config.context_length (throws std::out_of_range
// otherwise, via runtime/transformer/embedding.cpp).
ForwardResult forward(const model::Model& model, const std::vector<int32_t>& ids);

}  // namespace rt::execution
