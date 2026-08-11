#include "ops.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace rt::ops {

void linear(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& y) {
    assert(x.shape.ndim == 2 && W.shape.ndim == 2 && y.shape.ndim == 2);
    const uint32_t rows = x.shape[0];
    const uint32_t in_features = x.shape[1];
    const uint32_t out_features = W.shape[1];
    assert(W.shape[0] == in_features);
    assert(y.shape[0] == rows && y.shape[1] == out_features);
    assert(b.data == nullptr || (b.shape.ndim == 1 && b.shape[0] == out_features));

    const float* xd = static_cast<const float*>(x.data);
    const float* wd = static_cast<const float*>(W.data);
    const float* bd = static_cast<const float*>(b.data);
    float* yd = static_cast<float*>(y.data);

    for (uint32_t r = 0; r < rows; ++r) {
        const float* xrow = xd + static_cast<size_t>(r) * in_features;
        float* yrow = yd + static_cast<size_t>(r) * out_features;
        for (uint32_t o = 0; o < out_features; ++o) {
            float acc = bd != nullptr ? bd[o] : 0.0f;
            for (uint32_t i = 0; i < in_features; ++i) {
                acc += xrow[i] * wd[static_cast<size_t>(i) * out_features + o];
            }
            yrow[o] = acc;
        }
    }
}

void softmax_rows(Tensor& x) {
    assert(x.shape.ndim == 2);
    const uint32_t rows = x.shape[0];
    const uint32_t cols = x.shape[1];
    float* xd = static_cast<float*>(x.data);

    for (uint32_t r = 0; r < rows; ++r) {
        float* row = xd + static_cast<size_t>(r) * cols;
        float max_val = -std::numeric_limits<float>::infinity();
        for (uint32_t c = 0; c < cols; ++c) max_val = std::max(max_val, row[c]);

        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            // exp(-inf - finite) == 0, exactly what a fully-masked-out row needs.
            float e = std::exp(row[c] - max_val);
            row[c] = e;
            sum += e;
        }
        // A row that's entirely -inf (shouldn't happen with a causal mask, since the
        // diagonal is always unmasked) would give sum==0; guard rather than divide by 0.
        if (sum > 0.0f) {
            for (uint32_t c = 0; c < cols; ++c) row[c] /= sum;
        }
    }
}

}  // namespace rt::ops
