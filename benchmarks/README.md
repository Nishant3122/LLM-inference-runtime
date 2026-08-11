# benchmarks/

Benchmark configs and recorded results (spec §26-28). See
[`docs/benchmarks.md`](../docs/benchmarks.md) for what gets measured and why.

`results/` is gitignored (see root `.gitignore`) — regenerate via `tools/benchmark`
once it exists (Phase 1+), don't commit raw run output. Commit summarized/plotted
results under a `reports/` subdirectory instead, once there's something to report.
