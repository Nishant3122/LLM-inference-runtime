// Token + learned positional embedding, matching reference/model.py:
//   x = tok_embedding[ids] + pos_embedding[positions]
#pragma once

#include <cstdint>
#include <vector>

#include "../core/tensor.h"

namespace rt::transformer {

// tok_embedding: [vocab_size, D], pos_embedding: [context_length, D],
// ids: T token ids (each < vocab_size), y: [T, D] (pre-allocated).
// Position for ids[t] is `position_offset + t` (default 0: positions [0, T), the
// Phase 1 behavior for a full prompt starting a sequence). Phase 2's decode_step
// passes the token's actual absolute position for a single-token call where t is
// always 0 but the real sequence position isn't.
// Throws std::out_of_range if any id is out of bounds or position_offset + T > context_length.
void embedding_forward(const Tensor& tok_embedding, const Tensor& pos_embedding,
                        const std::vector<int32_t>& ids, Tensor& y, uint32_t position_offset = 0);

}  // namespace rt::transformer
