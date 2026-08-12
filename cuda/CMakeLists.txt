# Phase 3: naive CUDA kernels + device memory + model upload. Every kernel here
# mirrors a runtime/transformer or runtime/execution CPU op exactly (see each .cuh's
# header comment) — no shared-memory tiling, no fusion, no KV cache yet. Optimize only
# after Phase 4 profiling says a specific kernel is the bottleneck (Engineering
# Principle 1). Only configured when the top-level CMakeLists.txt has BUILD_CUDA=ON.
add_library(cuda_kernels STATIC
    utils/cuda_check.h

    memory/device_buffer.h
    memory/cuda_model.h
    memory/model_upload.cu

    kernels/linear.cuh
    kernels/linear.cu
    kernels/softmax.cuh
    kernels/softmax.cu
    kernels/layer_norm.cuh
    kernels/layer_norm.cu
    kernels/elementwise.cuh
    kernels/elementwise.cu
    kernels/embedding.cuh
    kernels/embedding.cu
    kernels/attention.cuh
    kernels/attention.cu
)
target_include_directories(cuda_kernels PUBLIC ${CMAKE_SOURCE_DIR})
# Deliberately NOT setting CUDA_SEPARABLE_COMPILATION (relocatable device code):
# no kernel here calls a __device__ function defined in another .cu file, so RDC
# buys nothing and just adds a device-link step every consumer needs to get right.
# (Turning it on caused "undefined reference to __cudaRegisterLinkedBinary_..." at
# link time for every target consuming this static library — removing it fixed it.)
