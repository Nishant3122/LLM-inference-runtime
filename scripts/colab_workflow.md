# CUDA development workflow: local edit -> Colab compile/run

This machine has no NVIDIA GPU and no CUDA Toolkit (`docs/architecture.md` §10), so
CUDA code (`cuda/`, `runtime/execution/cuda_backend.*`) is written here but compiled
and run on a free Google Colab T4 GPU instance. Used for all of Phase 3; expect to
reuse it for Phase 4 (CUDA optimization) and any later CUDA work.

## Setup (one-time per Colab session)

1. Open a new notebook at `colab.research.google.com`.
2. Runtime -> Change runtime type -> **T4 GPU** (free tier).
3. First cell — environment check + initial clone/build:
   ```python
   !nvidia-smi
   !nvcc --version
   !git clone https://github.com/Nishant3122/LLM-inference-runtime.git
   %cd LLM-inference-runtime
   !cmake -S . -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
   !cmake --build build -j
   !ctest --test-dir build --output-on-failure
   ```

## Iteration loop (every fix)

1. Edit locally, `git commit` + `git push` (this repo is public — Colab clones it
   with no auth needed).
2. Re-run this cell in the existing Colab session (full rebuild each time —
   `rm -rf build` avoids any stale-cache surprises; the whole project builds in well
   under a minute, so this hasn't been worth optimizing to an incremental build):
   ```python
   %cd /content/LLM-inference-runtime
   !git pull
   !rm -rf build
   !cmake -S . -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
   !cmake --build build -j
   !ctest --test-dir build --output-on-failure
   ```
3. Read the output, fix the next error locally, repeat.

## Getting model.bin onto Colab

`models/*.bin` are gitignored (reproducible artifacts, see `models/README.md`), so a
fresh Colab clone won't have them. Two options:
- **Regenerate on Colab** (what Phase 3 did): `reference/train.py` auto-selects
  `cuda` if available, so training runs *on the T4* and is dramatically faster than
  this machine's CPU (300 steps: 25s on Colab's T4 vs. ~1275s extrapolated from this
  machine's 1200-step/3193s run — roughly 50x). A short run (200-300 steps) is enough
  for a real trained model to test against; doesn't need to match the full 1200-step
  checkpoint used for `tests/golden/`.
  ```python
  %cd /content/LLM-inference-runtime/reference
  !python data/make_corpus.py
  !python train.py --steps 300 --eval-every 100
  !python export.py
  !python dump_golden.py
  %cd /content/LLM-inference-runtime
  ```
- **Upload directly** if you specifically need the *same* checkpoint local tests use
  (e.g. bit-exact comparison against `tests/golden/`) — via Colab's file upload UI or
  `google.colab.files.upload()`. Not needed for kernel-level or full-forward
  correctness checks, which just need *a* trained model, not a specific one.

## Driving this from Claude Code

Via the `claude-in-chrome` MCP tools (drives your actual logged-in Chrome — Claude
never touches Google credentials directly): open the notebook, click into a cell,
`type` the Python/shell lines, `ctrl+Return` to run in place, `get_page_text` to read
output (a screenshot doesn't show cell output text reliably — read the page instead).
`rm -rf build` means each round-trip is a full rebuild (~30-60s for this project's
size) — acceptable at this scale, revisit if the CUDA source tree grows enough that
iteration speed starts to hurt.
