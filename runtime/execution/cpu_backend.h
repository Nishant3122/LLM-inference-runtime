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

#include "../cache/kv_cache.h"
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
// otherwise, via runtime/transformer/embedding.cpp). Phase 1 baseline: no cache, full
// recompute every call — see runtime/execution/README.md for measured cost.
ForwardResult forward(const model::Model& model, const std::vector<int32_t>& ids);

// Phase 2: same computation as forward(), but also seeds `kv` with every layer's K/V
// for `ids` (typically the prompt, starting a fresh sequence at position 0). Use this
// once per sequence, then decode_step() for every token after.
ForwardResult prefill(const model::Model& model, const std::vector<int32_t>& ids,
                       cache::KVCache& kv);

// Phase 2: run exactly one new token through the model using `kv`, which must already
// hold every prior position's K/V (from a preceding prefill()/decode_step() call).
// `position` is this token's absolute index in the sequence — decode_step() only ever
// sees one token, so it can't infer position from an ids vector the way forward()/
// prefill() do; the caller (which is tracking kv.length()) must supply it.
// Returns just this token's logits ([vocab_size]) — a decode step never needs any
// other position's output.
std::vector<float> decode_step(const model::Model& model, int32_t token_id, uint32_t position,
                                cache::KVCache& kv);

}  // namespace rt::execution
