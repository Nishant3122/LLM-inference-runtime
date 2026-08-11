// MLP block matching reference/model.py::MLP: fc2(GELU(fc1(x))). Exact (erf-based)
// GELU, matching PyTorch's default F.gelu (no tanh approximation) — see
// runtime/transformer/ops.h.
#pragma once

#include "../core/tensor.h"

namespace rt::transformer {

struct MlpWeights {
    Tensor fc1_w, fc1_b;  // fc1_w: [D, d_ff], fc1_b: [d_ff]
    Tensor fc2_w, fc2_b;  // fc2_w: [d_ff, D], fc2_b: [D]
};

// x: [T, D] -> y: [T, D] (pre-allocated).
void mlp_forward(const Tensor& x, const MlpWeights& w, Tensor& y);

}  // namespace rt::transformer
