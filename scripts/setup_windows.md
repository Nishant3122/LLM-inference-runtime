# Dev environment setup (Windows)

Recorded here because `docs/architecture.md` §10 notes the primary dev machine is
currently missing all of this. Run these yourself when you're ready for Phase 1/3 —
none of this has been run automatically as part of Phase 0.

## Phase 1 (C++ CPU runtime): CMake + a C++17/20 compiler

Pick one compiler toolchain:

**Option A — MSVC (Visual Studio Build Tools):**
```powershell
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

**Option B — MinGW-w64 (lighter weight, no Visual Studio):**
```powershell
winget install Kitware.CMake
winget install MSYS2.MSYS2
# then, inside an MSYS2 MinGW64 shell:
# pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

Verify:
```powershell
cmake --version
```

Then configure/build (once `runtime/` has real Phase 1 sources beyond the Phase 0
`core/` skeleton):
```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Phase 3+ (CUDA backend): NVIDIA GPU + CUDA Toolkit

This machine has no `nvidia-smi`, i.e. no NVIDIA GPU/driver detected at all — CUDA
Toolkit installation alone won't be enough. Options when Phase 3 starts:

- Run on a different machine that has an NVIDIA GPU + driver installed.
- Use a cloud GPU instance (e.g. a single T4/L4/A10 instance is more than enough for
  the Stage-1 tiny model — no need for anything large).

Once on a machine with a GPU + driver:
```powershell
nvidia-smi                 # confirm driver + GPU are visible
# install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
nvcc --version              # confirm toolkit install
cmake -S . -B build -DBUILD_CUDA=ON
cmake --build build
```
