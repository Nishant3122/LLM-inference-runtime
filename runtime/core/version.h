// Single source of truth for the runtime version string, so `docs/observability.md`
// (spec §35, "reproducible benchmarks", Engineering Principle 5) can eventually print
// it alongside every benchmark result.
#pragma once

namespace rt {

const char* runtime_version();

}  // namespace rt
