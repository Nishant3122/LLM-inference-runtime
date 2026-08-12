# Placeholder so `cmake -DBUILD_CUDA=ON` configures cleanly before any real kernels
# exist (Phase 3). Once cuda/kernels has real .cu files, this becomes a target (e.g.
# `add_library(cuda_kernels STATIC kernels/matmul.cu ...)`) that runtime/execution's
# cuda_backend links against. Kept genuinely empty rather than a stub library with no
# sources — an empty add_library() is a CMake error, and there's nothing real to build
# yet (Engineering Principle 4: avoid premature abstraction).
