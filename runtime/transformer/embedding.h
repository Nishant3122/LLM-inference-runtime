// Token + learned positional embedding, matching reference/model.py:
//   x = tok_embedding[ids] + pos_embedding[positions]
#pragma once

#include <cstdint>
#include <vector>

#include "../core/tensor.h"

namespace rt::transformer {

// tok_embedding: [vocab_size, D], pos_embedding: [context_length, D],
// ids: T token ids (each < vocab_size), y: [T, D] (pre-allocated).
// Throws std::out_of_range if any id is out of bounds or T > context_length.
void embedding_forward(const Tensor& tok_embedding, const Tensor& pos_embedding,
                        const std::vector<int32_t>& ids, Tensor& y);

}  // namespace rt::transformer
