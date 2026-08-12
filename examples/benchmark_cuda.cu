// CUDA counterpart to examples/benchmark.cpp — same seq lengths, same profiler, same
// output format, so the two are directly diffable by eye. Only built when
// BUILD_CUDA=ON.
//
// Usage: benchmark_cuda [model.bin] [iterations]
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
        rt::profiling::Profiler profiler;
        for (int i = 0; i < iterations; ++i) {
            rt::execution::forward_cuda(model, ids, &profiler);
        }
        char title[64];
        std::snprintf(title, sizeof(title), "T=%u (%d iterations)", T, iterations);
        profiler.print_summary(std::cout, title);
        std::printf("\n");
    }

    return 0;
}
