// Top-k sampling: restrict to the k highest-probability tokens (after temperature
// scaling), then sample. Matches reference/generate.py's temperature>0, top_k>0
// branch.
#pragma once

#include <cstdint>
#include <random>

namespace rt::sampling {

int32_t top_k_sample(const float* logits, int vocab_size, int k, float temperature,
                      std::mt19937& rng);

}  // namespace rt::sampling
