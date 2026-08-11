#include "embedding.h"

#include <cassert>
#include <stdexcept>

namespace rt::transformer {

void embedding_forward(const Tensor& tok_embedding, const Tensor& pos_embedding,
                        const std::vector<int32_t>& ids, Tensor& y) {
    assert(tok_embedding.shape.ndim == 2 && pos_embedding.shape.ndim == 2);
    const uint32_t vocab_size = tok_embedding.shape[0];
    const uint32_t D = tok_embedding.shape[1];
    const uint32_t context_length = pos_embedding.shape[0];
    assert(pos_embedding.shape[1] == D);
    const uint32_t T = static_cast<uint32_t>(ids.size());
    assert(y.shape[0] == T && y.shape[1] == D);

    if (T > context_length) {
        throw std::out_of_range("embedding_forward: sequence length " + std::to_string(T) +
                                 " exceeds context_length " + std::to_string(context_length));
    }

    const float* tok = static_cast<const float*>(tok_embedding.data);
    const float* pos = static_cast<const float*>(pos_embedding.data);
    float* yd = static_cast<float*>(y.data);

    for (uint32_t t = 0; t < T; ++t) {
        int32_t id = ids[t];
        if (id < 0 || static_cast<uint32_t>(id) >= vocab_size) {
            throw std::out_of_range("embedding_forward: token id " + std::to_string(id) +
                                     " out of range [0, " + std::to_string(vocab_size) + ")");
        }
        const float* tok_row = tok + static_cast<size_t>(id) * D;
        const float* pos_row = pos + static_cast<size_t>(t) * D;
        float* out_row = yd + static_cast<size_t>(t) * D;
        for (uint32_t d = 0; d < D; ++d) out_row[d] = tok_row[d] + pos_row[d];
    }
}

}  // namespace rt::transformer
