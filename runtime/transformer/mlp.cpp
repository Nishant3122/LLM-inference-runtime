#include "mlp.h"

#include <cassert>
#include <vector>

#include "ops.h"

namespace rt::transformer {

void mlp_forward(const Tensor& x, const MlpWeights& w, Tensor& y) {
    assert(x.shape.ndim == 2);
    const uint32_t T = x.shape[0];
    const uint32_t D = x.shape[1];
    const uint32_t d_ff = w.fc1_w.shape[1];
    assert(w.fc1_w.shape[0] == D);
    assert(w.fc2_w.shape[0] == d_ff && w.fc2_w.shape[1] == D);
    assert(y.shape[0] == T && y.shape[1] == D);

    std::vector<float> hidden(static_cast<size_t>(T) * d_ff);
    Tensor hidden_t(hidden.data(), Shape{T, d_ff}, DataType::FP32, Device::CPU);
    ops::linear(x, w.fc1_w, w.fc1_b, hidden_t);

    for (float& v : hidden) v = ops::gelu(v);

    ops::linear(hidden_t, w.fc2_w, w.fc2_b, y);
}

}  // namespace rt::transformer
