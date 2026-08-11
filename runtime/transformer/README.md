# runtime/transformer — Phase 1

Not yet implemented. Will contain the CPU forward-pass ops, mirroring
[`reference/model.py`](../../reference/model.py) op-for-op so Phase 1's correctness
check (`docs/architecture.md` §4, G1) can bisect a mismatch to a single op:

- `embedding` — token + learned positional embedding lookup/add.
- `attention` — naive `softmax(QK^T / sqrt(head_dim)) V` with causal mask, matching
  `reference/model.py::CausalSelfAttention` exactly before any optimization.
- `mlp` — two linear layers + GELU, matching `reference/model.py::MLP`.
- `normalization` — LayerNorm.
- `transformer_block` — wires the above into the pre-norm block
  (`x = x + Attn(LN(x))`, `x = x + MLP(LN(x))`).

Implement naive first (spec §11: "This should first be implemented naively. Only
after correctness is established should optimization begin."). Validate each op
against the corresponding golden activation dump in `tests/golden/` (see
`reference/dump_golden.py`) before moving to the next.
