// CUDA error-checking macro. Every CUDA runtime call in this project should be
// wrapped in this — silently ignoring a cudaError_t is how you end up debugging a
// wrong answer three kernels downstream of the actual failure.
#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace rt::cuda {

inline void check_cuda(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << ": " << expr << " -> "
            << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

}  // namespace rt::cuda

#define CUDA_CHECK(expr) ::rt::cuda::check_cuda((expr), #expr, __FILE__, __LINE__)

// Call after launching a kernel to catch both launch-time errors (bad config) and
// anything the kernel itself faulted on, synchronously, at the call site — not three
// kernels later when some unrelated cudaMemcpy happens to surface the sticky error.
#define CUDA_CHECK_LAST_ERROR()                                            \
    do {                                                                   \
        CUDA_CHECK(cudaGetLastError());                                    \
        CUDA_CHECK(cudaDeviceSynchronize());                               \
    } while (0)
