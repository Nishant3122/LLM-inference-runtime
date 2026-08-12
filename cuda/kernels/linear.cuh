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

// Phase 4: fused linear with an optional epilogue, so a caller that needs
// linear -> GELU or linear -> residual-add doesn't pay for two extra kernel
// launches (and, for the residual case, an extra global-memory round trip) on top
// of this one. Added after profiling (cuda/README.md "Phase 4") showed launch count
// — not per-kernel compute — dominates at this model's scale: `attention` was 7
// launches per call, `mlp` was 3 + a separate residual add.
//
//   val = dot(x_row, W_col) + bias[col]
//   if (apply_gelu) val = gelu(val)
//   if (residual != nullptr) val += residual[same index]
//   y[idx] = val
//
// `y` and `residual` may point to the *same* buffer (in-place accumulation is safe:
// each thread only ever reads/writes its own output index, no cross-thread
// dependency) — this is exactly how the fused residual-add is used, see
// causal_self_attention_fused and cuda_backend.cu's fused MLP path.
void linear_fused(const rt::Tensor& x, const rt::Tensor& W, const rt::Tensor& b, rt::Tensor& y,
                   bool apply_gelu, const rt::Tensor* residual = nullptr);

}  // namespace rt::cuda::ops
