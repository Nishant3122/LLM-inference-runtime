// Elementwise ops: exact (erf-based) GELU, matching runtime/transformer/ops.h::gelu,
// and residual add (x += y), used by cuda_backend's transformer block.
#pragma once

#include "../../runtime/core/tensor.h"

namespace rt::cuda::ops {

// x: device tensor, any shape (treated as flat n = numel()). In place.
void gelu_inplace(rt::Tensor& x);

// x += y, elementwise, in place on x. Both device tensors, same numel().
void add_inplace(rt::Tensor& x, const rt::Tensor& y);

}  // namespace rt::cuda::ops
