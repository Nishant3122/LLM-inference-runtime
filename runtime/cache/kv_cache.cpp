#include "kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace rt::cache {

void LayerKVCache::init(uint32_t context_length, uint32_t d_model) {
    context_length_ = context_length;
    d_model_ = d_model;
    length_ = 0;
    k_.assign(static_cast<size_t>(context_length) * d_model, 0.0f);
    v_.assign(static_cast<size_t>(context_length) * d_model, 0.0f);
}

void LayerKVCache::append(const float* new_k, const float* new_v, uint32_t num_new) {
    if (length_ + num_new > context_length_) {
        throw std::out_of_range(
            "LayerKVCache::append: would exceed context_length (" +
            std::to_string(length_) + " + " + std::to_string(num_new) + " > " +
            std::to_string(context_length_) +
            "); Phase 2 does not implement sliding-window eviction, see runtime/cache/README.md");
    }
    size_t offset = static_cast<size_t>(length_) * d_model_;
    size_t nbytes = static_cast<size_t>(num_new) * d_model_ * sizeof(float);
    std::memcpy(k_.data() + offset, new_k, nbytes);
    std::memcpy(v_.data() + offset, new_v, nbytes);
    length_ += num_new;
}

void LayerKVCache::reset() { length_ = 0; }

void KVCache::init(uint32_t n_layers, uint32_t context_length, uint32_t d_model) {
    layers_.assign(n_layers, LayerKVCache{});
    for (auto& l : layers_) l.init(context_length, d_model);
}

void KVCache::reset() {
    for (auto& l : layers_) l.reset();
}

}  // namespace rt::cache
