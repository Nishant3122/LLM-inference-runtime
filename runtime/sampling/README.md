# runtime/sampling — Phase 1

Not yet implemented. Will contain `greedy`, `temperature`, `top_k` sampling
(spec §21), deliberately separated from the Transformer execution engine (they operate
on the final logits tensor only). Mirrors `reference/generate.py`'s three sampling
modes, which should be treated as the reference behavior to match.
