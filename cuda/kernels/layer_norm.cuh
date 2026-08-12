// Naive LayerNorm. Mirrors runtime/transformer/normalization.h::layer_norm (biased
// variance, eps=1e-5 default, matching torch.nn.LayerNorm).
#pragma once

#include "../../runtime/core/tensor.h"

namespace rt::cuda::transformer {

// x: [T, D], weight/bias: [D], y: [T, D] — all device tensors. One thread per row
// (T rows), matching the CPU version's per-row independence; double-precision
// accumulation for mean/var, same reason the CPU version uses it (tighter match
// against the PyTorch reference, which also accumulates at higher effective
// precision for a reduction this small).
void layer_norm(const rt::Tensor& x, const rt::Tensor& weight, const rt::Tensor& bias,
                 rt::Tensor& y, float eps = 1e-5f);

}  // namespace rt::cuda::transformer
