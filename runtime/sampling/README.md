# runtime/sampling — Phase 1 ✅

`greedy`, `temperature`, `top_k` sampling (spec §21), operating on a raw logits
pointer only — no dependency on the Transformer execution engine, matching
`reference/generate.py`'s three sampling modes exactly (temperature<=0 -> greedy;
temperature>0 & top_k<=0 -> temperature; temperature>0 & top_k>0 -> top-k). Used by
`examples/generate.cpp`'s autoregressive loop.
