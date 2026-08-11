// Fixed-rank shape (max 4 dims), matching the tensor table entry in
// docs/model_format.md (shape[4], ndim<=4) so a Tensor loaded from model.bin needs no
// conversion. Revisit the rank-4 cap if a later stage needs higher-rank tensors
// (e.g. batched KV cache might want [layer, batch, head, seq, head_dim] — cross that
// bridge in Phase 2, don't generalize now).
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rt {

struct Shape {
    static constexpr int kMaxDims = 4;

    std::array<uint32_t, kMaxDims> dims{0, 0, 0, 0};
    int ndim = 0;

    Shape() = default;

    Shape(std::initializer_list<uint32_t> d) {
        if (d.size() > static_cast<size_t>(kMaxDims)) {
            throw std::invalid_argument("Shape: more than kMaxDims dimensions");
        }
        ndim = static_cast<int>(d.size());
        int i = 0;
        for (auto v : d) dims[i++] = v;
    }

    uint32_t operator[](int i) const { return dims[i]; }

    uint64_t numel() const {
        uint64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= dims[i];
        return ndim == 0 ? 0 : n;
    }

    std::string to_string() const {
        std::string s = "[";
        for (int i = 0; i < ndim; ++i) {
            s += std::to_string(dims[i]);
            if (i + 1 < ndim) s += ", ";
        }
        s += "]";
        return s;
    }

    bool operator==(const Shape& other) const {
        if (ndim != other.ndim) return false;
        for (int i = 0; i < ndim; ++i) {
            if (dims[i] != other.dims[i]) return false;
        }
        return true;
    }
    bool operator!=(const Shape& other) const { return !(*this == other); }
};

}  // namespace rt
