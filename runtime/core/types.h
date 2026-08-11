// Core enums for the Tensor abstraction (docs/architecture.md §6, spec §8).
//
// Kept intentionally small: only what Phase 0/1 actually need. INT8/INT4 dtype tags
// exist now (so the Tensor/model-loader code doesn't need to change shape later) but
// have no arithmetic support until Phase 5 (quantization).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rt {

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
};

enum class Device : uint32_t {
    CPU = 0,
    CUDA = 1,
};

enum class Layout : uint32_t {
    RowMajor = 0,
    ColumnMajor = 1,
    Custom = 2,
};

// Bytes per element. INT4 is sub-byte (packed 2-per-byte); callers must not treat it
// as an addressable element size the way FP32/FP16/INT8 are — this returns 0 for INT4
// as a signal to use dtype-specific packing logic instead of pointer arithmetic.
inline std::size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT4: return 0;  // packed; see quantization/int4
    }
    throw std::invalid_argument("unknown DataType");
}

inline const char* to_string(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return "FP32";
        case DataType::FP16: return "FP16";
        case DataType::INT8: return "INT8";
        case DataType::INT4: return "INT4";
    }
    return "UNKNOWN";
}

inline const char* to_string(Device device) {
    switch (device) {
        case Device::CPU: return "CPU";
        case Device::CUDA: return "CUDA";
    }
    return "UNKNOWN";
}

}  // namespace rt
