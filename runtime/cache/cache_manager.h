// Deliberately thin for Stage 1 (single sequence at a time, no batching): a real
// cache_manager doing admission control (spec §33: "can this request's memory
// estimate fit?"), multi-sequence pooling, and eviction only earns its complexity
// once Phase 6 (batching) has multiple concurrent sequences to manage — see
// runtime/scheduler/README.md. For now this is just the one place that knows how to
// size a KVCache and estimate its footprint, so that logic isn't duplicated at every
// call site (and so Phase 7's memory estimator has something to build on).
#pragma once

#include <cstddef>
#include <cstdint>

#include "kv_cache.h"

namespace rt::cache {

// Bytes a KVCache with these dims will allocate: 2 (K and V) * n_layers *
// context_length * d_model * sizeof(float).
size_t estimate_kv_cache_bytes(uint32_t n_layers, uint32_t context_length, uint32_t d_model);

// Allocates and returns a ready-to-use KVCache (equivalent to `KVCache kv;
// kv.init(...); return kv;` — a named function mainly so a future admission check
// has one call site to add itself to, per the file comment above).
KVCache create_cache(uint32_t n_layers, uint32_t context_length, uint32_t d_model);

}  // namespace rt::cache
