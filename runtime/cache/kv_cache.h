// Per-layer, per-sequence K/V cache (spec §12). Stage 1 has one sequence generating at
// a time (no batching yet — Phase 6), so this is deliberately a single growable
// buffer per layer, not a pool shared across concurrent requests.
//
// Storage layout matches how runtime/transformer/attention.cpp already treats K/V:
// [context_length, d_model], heads as contiguous d_model/n_heads-sized chunks of the
// last dimension. Preallocated to context_length rows at init() so decode steps never
// reallocate (spec §13: avoid repeated allocation during token generation) — the one
// allocation happens once, at the start of a sequence.
#pragma once

#include <cstdint>
#include <vector>

namespace rt::cache {

class LayerKVCache {
public:
    void init(uint32_t context_length, uint32_t d_model);

    // Appends `num_new` rows of K and V (each [num_new, d_model], row-major) at the
    // current length, advancing it. Throws std::out_of_range if this would exceed
    // capacity (context_length) — Phase 2 does not implement sliding-window eviction;
    // see runtime/cache/README.md.
    void append(const float* new_k, const float* new_v, uint32_t num_new);

    void reset();  // length -> 0; keeps allocated capacity (no reallocation)

    uint32_t length() const { return length_; }
    uint32_t capacity() const { return context_length_; }
    uint32_t d_model() const { return d_model_; }

    // Valid prefix is [0, length()) rows; row stride is d_model().
    const float* keys() const { return k_.data(); }
    const float* values() const { return v_.data(); }

private:
    std::vector<float> k_, v_;  // each sized [context_length, d_model]
    uint32_t context_length_ = 0;
    uint32_t d_model_ = 0;
    uint32_t length_ = 0;
};

// One LayerKVCache per transformer layer.
class KVCache {
public:
    void init(uint32_t n_layers, uint32_t context_length, uint32_t d_model);

    LayerKVCache& layer(uint32_t i) { return layers_.at(i); }
    const LayerKVCache& layer(uint32_t i) const { return layers_.at(i); }
    uint32_t num_layers() const { return static_cast<uint32_t>(layers_.size()); }

    // All layers advance together (one token -> one K/V row per layer per step), so
    // layer 0's length is representative of the whole cache's.
    uint32_t length() const { return layers_.empty() ? 0 : layers_[0].length(); }

    void reset();

private:
    std::vector<LayerKVCache> layers_;
};

}  // namespace rt::cache
