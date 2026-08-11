# tools/benchmark — Phase 1+

Not yet implemented. Will be the CLI that runs the benchmark matrix described in
[`docs/benchmarks.md`](../../docs/benchmarks.md) and writes results into
`benchmarks/results/` (gitignored — see root `.gitignore`) in a format the ablation
studies (spec §32) and final comparison table (spec §43) can consume directly.
`examples/benchmark.cpp` is a thin example entry point; this is where the real
sweep/reporting logic will live once there's a runtime to benchmark.
