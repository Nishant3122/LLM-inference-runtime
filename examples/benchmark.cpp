// Example: run the benchmark matrix from docs/architecture.md / spec §28 (batch size
// x sequence length x precision x backend) and print TTFT/TPOT/throughput/memory
// (spec §27).
//
// TODO(Phase 1+): implement once runtime/profiling exists and there's a runtime to
// benchmark. Placeholder for now, see examples/generate.cpp for why.
#include <cstdio>

#include "../runtime/core/version.h"

int main() {
    std::printf("adaptive-llm-runtime %s\n", rt::runtime_version());
    std::printf("examples/benchmark: not yet implemented (Phase 1+ — see benchmarks/README.md)\n");
    return 0;
}
