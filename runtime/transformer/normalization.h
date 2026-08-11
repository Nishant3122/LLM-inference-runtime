// LayerNorm, matching PyTorch's nn.LayerNorm defaults: biased (population) variance
// over the last dimension, eps=1e-5 added inside the sqrt.
#pragma once

#include "../core/tensor.h"

namespace rt::transformer {

// x: [T, D], weight/bias: [D], y: [T, D] (pre-allocated). y and x may be the same
// buffer (element (t,d) of y only ever depends on row t of x).
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& y,
                 float eps = 1e-5f);

}  // namespace rt::transformer
