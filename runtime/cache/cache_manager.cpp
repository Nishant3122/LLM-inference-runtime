#include "cache_manager.h"

namespace rt::cache {

size_t estimate_kv_cache_bytes(uint32_t n_layers, uint32_t context_length, uint32_t d_model) {
    return 2ull * n_layers * context_length * d_model * sizeof(float);
}

KVCache create_cache(uint32_t n_layers, uint32_t context_length, uint32_t d_model) {
    KVCache kv;
    kv.init(n_layers, context_length, d_model);
    return kv;
}

}  // namespace rt::cache
