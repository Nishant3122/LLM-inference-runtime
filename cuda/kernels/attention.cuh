// Naive CUDA causal self-attention. Mirrors runtime/transformer/attention.h's
// no-cache path (causal_self_attention with kv_out == nullptr) exactly — same
// formula, same op decomposition (Q/K/V via linear, scores, causal mask, softmax,
// weighted sum, output projection).
//
// Reuses rt::transformer::AttentionWeights directly rather than defining a parallel
// CUDA-specific struct: Tensor already carries a Device tag, so the same struct
// works for both backends — model_upload.h just populates it with device pointers
// instead of host ones.
//
// No KV cache yet: this is the Phase 3 baseline (full O(T^2) recompute), the CUDA
// equivalent of Phase 1's cpu_backend::forward(). A CUDA-side KV cache (mirroring
// Phase 2's runtime/cache) is deliberately deferred — see cuda/README.md.
#pragma once

#include "../../runtime/core/tensor.h"
#include "../../runtime/transformer/attention.h"

namespace rt::cuda::transformer {

// x, y: device tensors [T, D]. w's Tensor fields must all be device tensors
// (populated by rt::cuda::upload_to_cuda).
void causal_self_attention(const rt::Tensor& x, const rt::transformer::AttentionWeights& w,
                            int n_heads, rt::Tensor& y);

}  // namespace rt::cuda::transformer
