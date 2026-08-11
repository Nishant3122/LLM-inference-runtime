# runtime/scheduler — Phase 6 (batching) / Phase 7 (adaptive)

Not yet implemented. Will contain:

- `request_scheduler` — admission (spec §33: can this request's memory estimate fit?),
  queueing.
- `batch_scheduler` — static batching first (Phase 6), then continuous/dynamic
  batching as a stretch goal (spec §18).
- `execution_policy` — the rule-based policy from spec §20:
  ```text
  if memory_pressure > threshold:      use INT8
  if batch_size > threshold:           use optimized batch kernel
  if sequence_length > threshold:      enable memory-efficient attention
  ```
  Policy is deliberately separate from mechanism (Engineering Principle 3): this
  module decides FP16/INT8/backend/batch size; `runtime/execution` implements the
  decision. A learned policy (spec §41) is an optional extension after the rule-based
  version has a measured A/B result to beat.
