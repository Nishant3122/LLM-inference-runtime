// CUDA backend forward pass. Reuses ForwardResult from cpu_backend.h unchanged — same
// fields, same shapes, all host-side — so tests can diff cuda output against cpu
// output with the exact same comparison code tests/model_test.cpp already uses.
//
// Only built when BUILD_CUDA=ON (see cuda/README.md, docs/architecture.md §10: this
// needs an NVIDIA GPU + CUDA Toolkit, not available on the primary dev machine).
#pragma once

#include <vector>

#include "../../cuda/memory/cuda_model.h"
#include "cpu_backend.h"

namespace rt::execution {

// Mirrors forward()'s no-cache behavior exactly (Phase 3 baseline, no CUDA-side KV
// cache yet — see cuda/README.md). ids.size() must not exceed
// model.config.context_length.
//
// `profiler`, if non-null (Phase 4), times the same coarse categories forward()
// does — see runtime/profiling/README.md. Each timed lambda ends with a device sync
// so this measures actual kernel execution, not just async launch overhead.
//
// `use_fusion` (Phase 4, default false): when true, uses the fused kernel variants
// (causal_self_attention_fused, linear_fused for MLP) that real profiling showed cut
// launch count where it mattered — see cuda/README.md "Phase 4". Default false keeps
// the exact Phase 3 baseline available for the A/B comparison Engineering Principle 2
// requires ("every optimization needs a baseline"); both paths compute the same
// values (verified in tests/cuda_ops_test.cu), just with fewer kernel launches when
// true.
ForwardResult forward_cuda(const cuda::CudaModel& model, const std::vector<int32_t>& ids,
                            profiling::Profiler* profiler = nullptr, bool use_fusion = false);

}  // namespace rt::execution
