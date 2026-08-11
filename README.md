# Adaptive LLM Inference Runtime

A lightweight, modular C++/CUDA inference runtime for decoder-only Transformers,
built independently of PyTorch at inference time. The project is a systems/GPU-computing
exercise, not a model-training exercise: the model is a means to study the runtime
(memory management, KV-caching, quantization, CUDA kernels, batching, and adaptive
scheduling), not an end in itself.

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/model_format.md`](docs/model_format.md) for the on-disk model format.

## Status

**Phase 0 — Architecture & Reference** (see [Development Phases](docs/architecture.md#development-phases))

| Phase | Status |
|---|---|
| 0 — Architecture & reference model | 🚧 in progress |
| 1 — CPU runtime | ⬜ not started |
| 2 — KV cache | ⬜ not started |
| 3 — CUDA backend | ⬜ not started |
| 4 — CUDA optimization | ⬜ not started |
| 5 — Quantization | ⬜ not started |
| 6 — Batching | ⬜ not started |
| 7 — Adaptive runtime | ⬜ not started |

## Target model (Stage 1)

A **tiny, custom, char-level decoder-only Transformer** — small enough to run on CPU
in milliseconds and to hand-verify individual matmuls:

- vocab: character-level (~70-100 symbols, built from the training corpus)
- 4 layers, `d_model=128`, 4 heads (head_dim=32), `d_ff=512`
- context length 256, learned absolute positional embeddings
- pre-norm (LayerNorm → Attention → residual, LayerNorm → MLP → residual)
- weights exported to a custom binary format (`docs/model_format.md`), independent of PyTorch

This is deliberately not GPT-2-class. See [`docs/architecture.md`](docs/architecture.md) §6
for the reasoning and the planned progression to larger models.

## Local environment note

This machine currently has **no NVIDIA GPU/driver, no CUDA toolkit, no CMake, and no
C++ compiler** — only Python + PyTorch(CPU) and Git. Phase 0 (reference model, model
format, docs) only needs Python. Building the C++ runtime (Phase 1+) needs CMake and a
C++17/20 compiler; the CUDA backend (Phase 3+) needs an NVIDIA GPU + CUDA Toolkit
(locally or via a cloud GPU instance).

## Quick start (Phase 0)

```bash
cd reference
pip install -r requirements.txt
python data/make_corpus.py          # generate the toy training corpus
python train.py                     # train the tiny reference model (CPU, seconds)
python generate.py                  # sanity-check text generation
python export.py                    # export weights -> ../models/model.bin
python dump_golden.py               # dump golden logits -> ../tests/golden/ for later runtime comparison
```

## Repository layout

```text
docs/            architecture, model format, memory model, scheduler, benchmark docs
reference/       PyTorch reference model, training, export to model.bin (source of truth for G1 correctness)
runtime/         C++ runtime (core, model, transformer, execution, cache, quantization, scheduler, sampling)
cuda/            CUDA kernels, memory, utils
tools/           model_converter, tokenizer, benchmark CLIs
tests/           unit/integration/numerical tests + tests/golden reference outputs
benchmarks/      benchmark configs and recorded results
scripts/         setup/dev scripts
models/          generated model.bin / vocab.json (gitignored, reproducible from reference/)
examples/        example programs (generate.cpp, benchmark.cpp)
```

## Non-goals (v1)

Training, distributed/multi-node inference, MoE, encoder-decoder architectures,
speculative decoding, multi-GPU tensor parallelism, production-grade serving,
full HuggingFace coverage, custom tokenizer training, full PyTorch compatibility.
See [`docs/architecture.md`](docs/architecture.md) §5.
