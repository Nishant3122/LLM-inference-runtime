// The Tensor abstraction, per docs/architecture.md §6 / spec §8.
//
// Deliberately a non-owning view for now: `data` is a raw pointer, no
// allocation/deallocation lives here. Ownership (host allocator, and later the CUDA
// memory pool from spec §13) is a Phase 1 concern — see runtime/model/weight_manager
// and, eventually, a dedicated allocator. Building that now would be the premature
// abstraction Engineering Principle 4 warns against: we don't yet know the real
// allocation patterns (weights loaded once vs. activation buffers reused every token).
#pragma once

#include "shape.h"
#include "types.h"

namespace rt {

struct Tensor {
    void* data = nullptr;
    Shape shape;
    DataType dtype = DataType::FP32;
    Device device = Device::CPU;
    Layout layout = Layout::RowMajor;

    Tensor() = default;
    Tensor(void* data_, Shape shape_, DataType dtype_, Device device_,
           Layout layout_ = Layout::RowMajor)
        : data(data_), shape(shape_), dtype(dtype_), device(device_), layout(layout_) {}

    uint64_t numel() const { return shape.numel(); }

    // Only meaningful for byte-addressable dtypes (FP32/FP16/INT8); INT4 is packed
    // 2-per-byte, see quantization/int4 once that lands (Phase 5).
    uint64_t nbytes() const { return numel() * dtype_size(dtype); }

    bool is_cpu() const { return device == Device::CPU; }
    bool is_cuda() const { return device == Device::CUDA; }
};

}  // namespace rt
