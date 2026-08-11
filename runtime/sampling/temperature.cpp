#include "temperature.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rt::sampling {

int32_t temperature_sample(const float* logits, int vocab_size, float temperature,
                            std::mt19937& rng) {
    std::vector<float> scaled(vocab_size);
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        scaled[i] = logits[i] / temperature;
        max_val = std::max(max_val, scaled[i]);
    }

    std::vector<double> probs(vocab_size);
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        double p = std::exp(static_cast<double>(scaled[i] - max_val));
        probs[i] = p;
        sum += p;
    }

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    (void)sum;  // discrete_distribution normalizes internally; kept for clarity/debuggability
    return dist(rng);
}

}  // namespace rt::sampling
