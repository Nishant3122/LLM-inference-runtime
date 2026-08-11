# runtime/transformer — Phase 1 ✅

CPU forward-pass ops, mirroring [`reference/model.py`](../../reference/model.py)
op-for-op so `tests/model_test.cpp` can bisect a mismatch to a single op:

- `embedding` — token + learned positional embedding lookup/add.
- `attention` — naive `softmax(QK^T / sqrt(head_dim)) V` with causal mask, matching
  `reference/model.py::CausalSelfAttention` exactly. `O(T^2)` scores matrix
  materialized in full, no KV cache (Phase 2), no fused kernel (Phase 4+).
- `mlp` — two linear layers + exact (erf-based) GELU, matching `reference/model.py::MLP`.
- `normalization` — LayerNorm (biased variance, eps=1e-5, matching `torch.nn.LayerNorm`).
- `transformer_block` — wires the above into the pre-norm block
  (`x = x + Attn(LN(x))`, `x = x + MLP(LN(x))`).
- `ops.h/.cpp` — shared naive primitives (`linear`/matmul, `softmax_rows`, `gelu`)
  used by `attention.cpp` and `mlp.cpp`. Not in the original spec's directory list,
  but both ops need "linear layer"; one tested implementation beats two.

All naive by design (spec §11): plain triple-loop matmul, no blocking/tiling/BLAS.
Validated against `tests/golden/` (see `reference/dump_golden.py`) in
`tests/model_test.cpp` — max abs diff vs. the PyTorch reference is ~1e-6 (FP32
summation-order noise, not a correctness gap). Optimize only after Phase 4 profiling
says a specific op is the bottleneck — this is deliberately not that yet.
