// CUDA counterpart to examples/benchmark.cpp — same seq lengths, same profiler, same
// output format, so the two are directly diffable by eye. Only built when
// BUILD_CUDA=ON.
//
// Phase 4: also runs naive (use_fusion=false) vs fused (use_fusion=true) back to
// back, two ways:
//   1. With a Profiler attached (per-op breakdown, but every timed section forces a
//      cudaDeviceSynchronize — see runtime/profiling/README.md — so the *absolute*
//      numbers here are slower than real usage, just useful for relative shares).
//   2. Unprofiled wall-clock total (profiler=nullptr, kernels run async, one sync at
//      the very end via the host download that already happens) — this is the
//      number that actually answers "did fusion help," not the profiled one.
//
// Usage: benchmark_cuda [model.bin] [iterations]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../cuda/memory/cuda_model.h"
#include "../runtime/core/version.h"
#include "../runtime/execution/cuda_backend.h"
#include "../runtime/model/model_loader.h"
#include "../runtime/profiling/profiler.h"

namespace {

std::vector<int32_t> make_ids(uint32_t T, uint32_t vocab_size) {
    std::vector<int32_t> ids(T);
    for (uint32_t i = 0; i < T; ++i) ids[i] = static_cast<int32_t>(i % vocab_size);
    return ids;
}

double wall_clock_ms(const rt::cuda::CudaModel& model, const std::vector<int32_t>& ids,
                      int iterations, bool use_fusion) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        rt::execution::forward_cuda(model, ids, /*profiler=*/nullptr, use_fusion);
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "models/model.bin";
    int iterations = argc > 2 ? std::atoi(argv[2]) : 5;

    std::printf("adaptive-llm-runtime %s (Phase 4 profiling: CUDA)\n", rt::runtime_version());

    rt::model::Model host_model;
    try {
        host_model = rt::model::load_model(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "(expected model.bin from `python reference/export.py`)\n");
        return 1;
    }
    rt::cuda::CudaModel model = rt::cuda::upload_to_cuda(host_model);
    std::printf("Model: vocab=%u d_model=%u n_layers=%u n_heads=%u d_ff=%u context_length=%u\n\n",
                model.config.vocab_size, model.config.d_model, model.config.n_layers,
                model.config.n_heads, model.config.d_ff, model.config.context_length);

    std::vector<uint32_t> seq_lengths = {8, 32, 128, 256};
    for (auto& t : seq_lengths) {
        if (t > model.config.context_length) t = model.config.context_length;
    }

    for (uint32_t T : seq_lengths) {
        std::vector<int32_t> ids = make_ids(T, model.config.vocab_size);

        rt::profiling::Profiler naive_profiler;
        for (int i = 0; i < iterations; ++i) {
            rt::execution::forward_cuda(model, ids, &naive_profiler, /*use_fusion=*/false);
        }
        char naive_title[80];
        std::snprintf(naive_title, sizeof(naive_title), "T=%u naive (%d iterations)", T,
                      iterations);
        naive_profiler.print_summary(std::cout, naive_title);
        std::printf("\n");

        rt::profiling::Profiler fused_profiler;
        for (int i = 0; i < iterations; ++i) {
            rt::execution::forward_cuda(model, ids, &fused_profiler, /*use_fusion=*/true);
        }
        char fused_title[80];
        std::snprintf(fused_title, sizeof(fused_title), "T=%u fused (%d iterations)", T,
                      iterations);
        fused_profiler.print_summary(std::cout, fused_title);
        std::printf("\n");

        double naive_ms = wall_clock_ms(model, ids, iterations, false);
        double fused_ms = wall_clock_ms(model, ids, iterations, true);
        std::printf("T=%u unprofiled wall-clock: naive=%.3fms fused=%.3fms speedup=%.2fx\n\n", T,
                    naive_ms, fused_ms, naive_ms / fused_ms);
    }

    return 0;
}
