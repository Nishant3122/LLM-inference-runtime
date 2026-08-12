# Adaptive LLM Inference Runtime

A lightweight, modular C++/CUDA inference runtime for decoder-only Transformers,
built independently of PyTorch at inference time. The project is a systems/GPU-computing
exercise, not a model-training exercise: the model is a means to study the runtime
(memory management, KV-caching, quantization, CUDA kernels, batching, and adaptive
scheduling), not an end in itself.

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/model_format.md`](docs/model_format.md) for the on-disk model format.

## Status

**Phase 3 — CUDA Backend (naive), done** (see [Development Phases](docs/architecture.md#development-phases))

| Phase | Status |
|---|---|
| 0 — Architecture & reference model | ✅ done |
| 1 — CPU runtime | ✅ done — see `runtime/`, validated by `tests/model_test.cpp` against `tests/golden/` |
| 2 — KV cache | ✅ done — see `runtime/cache/`; cached vs. recomputed logits match exactly (max abs diff = 0); **24.6x**–**72.3x** measured speedup, see `runtime/cache/README.md` |
| 3 — CUDA backend | ✅ done — see `cuda/`; every kernel + the full forward pass verified against CPU on a real Tesla T4 (max abs diff ~1e-6), see `cuda/README.md`. No CUDA-side KV cache yet |
| 4 — CUDA optimization | ⬜ next |
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

CMake 4.4.2 and MSVC Build Tools 2022 are installed and verified — the full CPU
runtime builds and passes `ctest` (see `docs/architecture.md` §10). This machine still
has no NVIDIA GPU/driver, so the CUDA backend (`cuda/`, `-DBUILD_CUDA=ON`) can't be
built or run here — it's developed locally and verified on a free Google Colab T4 GPU
instead; see `scripts/colab_workflow.md`.

## Quick start

**1. Train the reference model and export it (Python, Phase 0):**

```bash
cd reference
pip install -r requirements.txt
python data/make_corpus.py          # generate the toy training corpus
python train.py                     # train the tiny reference model (CPU, ~1hr for 1200 steps)
python generate.py                  # sanity-check text generation
python export.py                    # export -> ../models/model.bin, ../models/vocab.bin
python dump_golden.py               # dump golden activations -> ../tests/golden/ for the C++ tests
```

**2. Build and run the C++ runtime (Phase 1):**

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config RelWithDebInfo
ctest --test-dir build -C RelWithDebInfo --output-on-failure   # tensor/tokenizer/model/kv_cache tests

.\build\examples\RelWithDebInfo\generate_example.exe models\model.bin models\vocab.bin "The cat" 200 0.8 20
```

No PyTorch/libtorch involved in that last step (G2, `docs/architecture.md` §4) — pure
C++ loading `model.bin` and running the forward pass itself. Generation uses a
persistent KV cache by default (Phase 2) — pass `--no-cache` as a trailing argument to
fall back to the Phase 1 recompute-everything path and compare timings yourself; see
`runtime/cache/README.md` for the measured **24.6x**–**72.3x** speedup.

**3. CUDA backend (Phase 3) — needs an NVIDIA GPU, e.g. a free Colab T4:**

```bash
cmake -S . -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure   # includes cuda_ops_test
```

See `scripts/colab_workflow.md` for the exact Colab setup this was developed and
verified against.

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
models/          generated model.bin / vocab.bin / vocab.json (gitignored, reproducible from reference/)
examples/        example programs (generate.cpp, benchmark.cpp)
```

## Non-goals (v1)

Training, distributed/multi-node inference, MoE, encoder-decoder architectures,
speculative decoding, multi-GPU tensor parallelism, production-grade serving,
full HuggingFace coverage, custom tokenizer training, full PyTorch compatibility.
See [`docs/architecture.md`](docs/architecture.md) §5.
