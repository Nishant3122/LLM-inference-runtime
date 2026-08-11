#include "greedy.h"

namespace rt::sampling {

int32_t greedy_sample(const float* logits, int vocab_size) {
    int32_t best_id = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return best_id;
}

}  // namespace rt::sampling
