#include "normalization.h"

#include <cassert>
#include <cmath>

namespace rt::transformer {

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& y, float eps) {
    assert(x.shape.ndim == 2);
    const uint32_t T = x.shape[0];
    const uint32_t D = x.shape[1];
    assert(weight.shape.ndim == 1 && weight.shape[0] == D);
    assert(bias.shape.ndim == 1 && bias.shape[0] == D);
    assert(y.shape[0] == T && y.shape[1] == D);

    const float* xd = static_cast<const float*>(x.data);
    const float* wd = static_cast<const float*>(weight.data);
    const float* bd = static_cast<const float*>(bias.data);
    float* yd = static_cast<float*>(y.data);

    for (uint32_t t = 0; t < T; ++t) {
        const float* row = xd + static_cast<size_t>(t) * D;
        float* out_row = yd + static_cast<size_t>(t) * D;

        double mean = 0.0;
        for (uint32_t d = 0; d < D; ++d) mean += row[d];
        mean /= D;

        double var = 0.0;
        for (uint32_t d = 0; d < D; ++d) {
            double diff = row[d] - mean;
            var += diff * diff;
        }
        var /= D;  // biased variance, matches torch.nn.LayerNorm

        float inv_std = static_cast<float>(1.0 / std::sqrt(var + eps));
        for (uint32_t d = 0; d < D; ++d) {
            float normalized = static_cast<float>((row[d] - mean)) * inv_std;
            out_row[d] = normalized * wd[d] + bd[d];
        }
    }
}

}  // namespace rt::transformer
