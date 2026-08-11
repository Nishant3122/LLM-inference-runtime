#include "top_k.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rt::sampling {

int32_t top_k_sample(const float* logits, int vocab_size, int k, float temperature,
                      std::mt19937& rng) {
    k = std::min(k, vocab_size);

    std::vector<float> scaled(vocab_size);
    for (int i = 0; i < vocab_size; ++i) scaled[i] = logits[i] / temperature;

    // Find the k-th largest value (partial sort by value, descending), matching
    // reference/generate.py's torch.topk + "logits[logits < kth_largest] = -inf".
    std::vector<float> sorted_desc = scaled;
    std::nth_element(sorted_desc.begin(), sorted_desc.begin() + (k - 1), sorted_desc.end(),
                      std::greater<float>());
    float kth_largest = sorted_desc[k - 1];

    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        if (scaled[i] < kth_largest) scaled[i] = -std::numeric_limits<float>::infinity();
        max_val = std::max(max_val, scaled[i]);
    }

    std::vector<double> probs(vocab_size, 0.0);
    for (int i = 0; i < vocab_size; ++i) {
        if (scaled[i] == -std::numeric_limits<float>::infinity()) continue;
        probs[i] = std::exp(static_cast<double>(scaled[i] - max_val));
    }

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

}  // namespace rt::sampling
