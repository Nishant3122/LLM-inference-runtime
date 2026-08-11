// Small naive math primitives shared by attention.cpp and mlp.cpp. Not part of the
// original spec's directory list, but attention and MLP both need "linear layer" and
// MLP needs GELU — better as one place to get right and test than duplicated.
// O(T*in*out) triple-loop matmul, no blocking/tiling/BLAS: correctness first (spec
// §11, Engineering Principle 1). Revisit in Phase 4 once profiling says this matters.
#pragma once

#include <cmath>

#include "../core/tensor.h"

namespace rt::ops {

// y = x @ W + b
// x: [rows, in_features], W: [in_features, out_features] (already the
// model.bin-transposed layout, see docs/model_format.md), b: [out_features] or an
// empty Tensor (data==nullptr) to skip the bias add, y: [rows, out_features]
// (pre-allocated).
void linear(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& y);

// In-place row-wise softmax: for each row of x ([rows, cols]), x[r] := softmax(x[r]).
// -inf entries (used for causal masking) are handled correctly (their weight -> 0).
void softmax_rows(Tensor& x);

// Exact GELU (erf-based), matching PyTorch's default F.gelu (no tanh approximation).
inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));  // 1/sqrt(2)
}

}  // namespace rt::ops
