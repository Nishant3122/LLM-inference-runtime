// Minimal, backend-agnostic profiler: label -> {call count, total time}. Deliberately
// not CUDA-aware itself (no cuda_runtime.h dependency, builds without BUILD_CUDA) —
// for a CUDA call to be timed accurately (kernel launches are async), the *caller*
// must include a device sync inside the timed lambda; see
// runtime/execution/cuda_backend.cu for the pattern.
//
// Exists for Phase 4 (spec §30: "Embedding, Attention, QKV projection, Softmax, MLP,
// Normalization... should have separate timing measurements") — "profile first,
// optimize second" (Engineering Principle 1) needs something to profile with.
// Opt-in only: every instrumented call site takes a `Profiler*` that defaults to
// nullptr, so uninstrumented (production) runs pay zero overhead.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace rt::profiling {

struct OpStats {
    uint64_t calls = 0;
    double total_ms = 0.0;

    double avg_ms() const { return calls > 0 ? total_ms / static_cast<double>(calls) : 0.0; }
};

class Profiler {
public:
    // Times `fn`, wall-clock, and accumulates under `label`. For GPU work, `fn` must
    // synchronize internally (e.g. call cudaDeviceSynchronize() at its end) or this
    // only measures async launch overhead, not actual kernel execution time.
    void time(const std::string& label, const std::function<void()>& fn) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto& stats = stats_[label];
        stats.calls += 1;
        stats.total_ms += ms;
    }

    void reset() { stats_.clear(); }

    const std::unordered_map<std::string, OpStats>& stats() const { return stats_; }

    // Sorted by total_ms descending, with each label's % of the summed total.
    void print_summary(std::ostream& os, const std::string& title = "") const;

private:
    std::unordered_map<std::string, OpStats> stats_;
};

// Times `fn` under `label` if `profiler` is non-null, otherwise just calls `fn()`
// directly. Lets an instrumented call site read the same either way instead of an
// if/else at every one — see runtime/transformer/transformer_block.cpp and
// runtime/execution/cuda_backend.cu.
inline void time_if(Profiler* profiler, const std::string& label,
                     const std::function<void()>& fn) {
    if (profiler != nullptr) {
        profiler->time(label, fn);
    } else {
        fn();
    }
}

}  // namespace rt::profiling
