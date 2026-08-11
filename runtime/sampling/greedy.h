// Greedy decoding: token = argmax(logits). Matches reference/generate.py's
// temperature<=0 branch.
#pragma once

#include <cstdint>

namespace rt::sampling {

int32_t greedy_sample(const float* logits, int vocab_size);

}  // namespace rt::sampling
