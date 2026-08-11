// Temperature sampling: P(token) ~ softmax(logits / temperature). Matches
// reference/generate.py's temperature>0, top_k<=0 branch.
#pragma once

#include <cstdint>
#include <random>

namespace rt::sampling {

int32_t temperature_sample(const float* logits, int vocab_size, float temperature,
                            std::mt19937& rng);

}  // namespace rt::sampling
