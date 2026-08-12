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

// Phase 4: identical math to causal_self_attention, but with two fusions motivated by
// real profiling (cuda/README.md "Phase 4 optimization") — at this model's scale,
// attention's 7 separate kernel launches (wq, wk, wv, scores, softmax, weighted-sum,
// wo) were themselves the bottleneck, not any single kernel's compute:
//   1. Q/K/V computed in one launch instead of three (see qkv_fused).
//   2. The output projection adds directly into `x_inout` (the residual stream)
//      instead of writing to a scratch buffer that a separate add_inplace call then
//      has to read back — one linear_fused call replaces wo + add_inplace.
// Net: 7 launches -> 5. `x_inout` is both the pre-attention residual input (added at
// the end) and the output (attention's contribution accumulated into it) — same
// in-place-accumulation safety argument as linear_fused's `residual` parameter.
void causal_self_attention_fused(const rt::Tensor& x_norm,
                                  const rt::transformer::AttentionWeights& w, int n_heads,
                                  rt::Tensor& x_inout);

}  // namespace rt::cuda::transformer
