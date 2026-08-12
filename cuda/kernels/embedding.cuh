// Token + positional embedding lookup. Mirrors
// runtime/transformer/embedding.h::embedding_forward exactly, including the
// position_offset parameter Phase 2 added for decode steps.
#pragma once

#include <cstdint>
#include <vector>

#include "../../runtime/core/tensor.h"

namespace rt::cuda::transformer {

// tok_embedding, pos_embedding, y must be device tensors. `ids` is a host vector
// (small — at most context_length long) uploaded internally to a scratch device
// buffer each call; mirrors the CPU signature so tests can call both with the same
// arguments.
void embedding_forward(const rt::Tensor& tok_embedding, const rt::Tensor& pos_embedding,
                        const std::vector<int32_t>& ids, rt::Tensor& y,
                        uint32_t position_offset = 0);

}  // namespace rt::cuda::transformer
