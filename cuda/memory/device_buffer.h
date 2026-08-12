// Minimal RAII device-memory buffer. Header-only (all CUDA runtime calls here are
// plain host-callable C functions, so this doesn't need its own .cu translation
// unit) — one cudaMalloc per buffer, freed on destruction, move-only so ownership is
// never ambiguous.
//
// Deliberately not a pool yet (spec §13's "avoid cudaMalloc/cudaFree during token
// generation" — the memory pool called out there). Phase 3's job is a correct naive
// CUDA backend; a real pool is Phase 4 territory once profiling shows allocation
// overhead actually matters at this model's scale. model_upload.h already sidesteps
// the worst of it for weights (one allocation, once, at model load) — it's per-step
// activation buffers in cuda_backend that would benefit most from pooling later.
#pragma once

#include <cstddef>
#include <utility>

#include "../utils/cuda_check.h"

namespace rt::cuda {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t nbytes) : nbytes_(nbytes) {
        if (nbytes_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, nbytes_));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);  // destructors must not throw
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), nbytes_(other.nbytes_) {
        other.data_ = nullptr;
        other.nbytes_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) cudaFree(data_);
            data_ = other.data_;
            nbytes_ = other.nbytes_;
            other.data_ = nullptr;
            other.nbytes_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t nbytes() const { return nbytes_; }

    // Byte offset into this buffer as a typed pointer (for building Tensor views).
    template <typename T>
    T* at(size_t byte_offset) {
        return reinterpret_cast<T*>(static_cast<char*>(data_) + byte_offset);
    }

    void upload(const void* host_src, size_t n, size_t device_offset = 0) {
        CUDA_CHECK(cudaMemcpy(static_cast<char*>(data_) + device_offset, host_src, n,
                               cudaMemcpyHostToDevice));
    }

    void download(void* host_dst, size_t n, size_t device_offset = 0) const {
        CUDA_CHECK(cudaMemcpy(host_dst, static_cast<const char*>(data_) + device_offset, n,
                               cudaMemcpyDeviceToHost));
    }

private:
    void* data_ = nullptr;
    size_t nbytes_ = 0;
};

}  // namespace rt::cuda
