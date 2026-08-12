// Naive CUDA linear layer (matmul + bias): y = x @ W + b. Mirrors
// runtime/transformer/ops.h::linear exactly (same shapes, same [in,out] weight
// layout from docs/model_format.md) so cuda_ops_test can diff the two directly.
//
// One thread per output element (rows * out_features total threads), each doing the
// full in_features-length dot product — no shared-memory tiling, no coalescing
// tuning. That's Phase 4, after profiling says this kernel is worth the complexity.
#pragma once

#include "../../runtime/core/tensor.h"

namespace rt::cuda::ops {

// x, W, b, y must all be device tensors (Tensor::device == Device::CUDA). Pass an
// empty Tensor (data == nullptr) for `b` to skip the bias add, same convention as the
// CPU version.
void linear(const rt::Tensor& x, const rt::Tensor& W, const rt::Tensor& b, rt::Tensor& y);

}  // namespace rt::cuda::ops
