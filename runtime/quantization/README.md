# runtime/quantization — Phase 5

Not yet implemented. Will contain `fp16`, `int8`, `int4` conversion/execution support
(`docs/architecture.md` §4 G6). Plan, per the original spec §16 and §45:

- FP16 first (simplest: no calibration, just narrower storage + matching arithmetic).
- INT8 next, starting with simple symmetric per-tensor quantization before moving to
  per-channel/group-wise if the accuracy loss on the tiny model warrants it.
- INT4 opportunistically, time permitting.

For each precision, measure (spec §16): model size, GPU memory, latency, tokens/sec,
and an accuracy proxy appropriate for a char-level model (next-char cross-entropy /
perplexity on a held-out slice of `reference/data/corpus.txt`, and generated-token
agreement against the FP32 reference — exact equality is not expected once quantized,
per `docs/architecture.md` §4 G1 caveat).
