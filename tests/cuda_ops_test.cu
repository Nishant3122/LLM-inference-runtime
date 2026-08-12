// Kernel-level correctness for Phase 3: every CUDA op vs. its CPU counterpart, on
// random synthetic data (no model.bin needed — this is a unit test of the kernels
// themselves, not the trained model). This is "naive CUDA -> correct CUDA" from spec
// §14's development sequence; only after these pass does optimization (Phase 4) make
// sense.
//
// Also runs a full forward_cuda() vs forward() comparison if models/model.bin exists
// (skipped, not failed, if it doesn't — that file is gitignored/reproducible, see
// models/README.md, and this test shouldn't require regenerating it on every machine
// just to check the kernels are right).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "../cuda/kernels/attention.cuh"
#include "../cuda/kernels/elementwise.cuh"
#include "../cuda/kernels/embedding.cuh"
#include "../cuda/kernels/layer_norm.cuh"
#include "../cuda/kernels/linear.cuh"
#include "../cuda/kernels/softmax.cuh"
#include "../cuda/memory/cuda_model.h"
#include "../cuda/memory/device_buffer.h"
#include "../runtime/execution/cpu_backend.h"
#include "../runtime/execution/cuda_backend.h"
#include "../runtime/model/model_loader.h"
#include "../runtime/transformer/attention.h"
#include "../runtime/transformer/embedding.h"
#include "../runtime/transformer/normalization.h"
#include "../runtime/transformer/ops.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

std::mt19937 g_rng(1337);

std::vector<float> random_vec(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(g_rng);
    return v;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Uploads a host vector to a fresh DeviceBuffer and returns a device Tensor view
// into it. Caller must keep `buf` alive as long as the returned Tensor is used.
rt::Tensor to_device(const std::vector<float>& host, rt::Shape shape, rt::cuda::DeviceBuffer& buf) {
    buf = rt::cuda::DeviceBuffer(host.size() * sizeof(float));
    buf.upload(host.data(), host.size() * sizeof(float));
    return rt::Tensor(buf.data(), shape, rt::DataType::FP32, rt::Device::CUDA);
}

std::vector<float> to_host(const rt::cuda::DeviceBuffer& buf, size_t n) {
    std::vector<float> host(n);
    buf.download(host.data(), n * sizeof(float));
    return host;
}

void test_linear(float atol) {
    const uint32_t T = 7, in_f = 16, out_f = 24;
    auto x_h = random_vec(T * in_f);
    auto w_h = random_vec(in_f * out_f);
    auto b_h = random_vec(out_f);

    rt::Tensor x_cpu(x_h.data(), rt::Shape{T, in_f}, rt::DataType::FP32, rt::Device::CPU);
    rt::Tensor w_cpu(w_h.data(), rt::Shape{in_f, out_f}, rt::DataType::FP32, rt::Device::CPU);
    rt::Tensor b_cpu(b_h.data(), rt::Shape{out_f}, rt::DataType::FP32, rt::Device::CPU);
    std::vector<float> y_cpu_h(T * out_f);
    rt::Tensor y_cpu(y_cpu_h.data(), rt::Shape{T, out_f}, rt::DataType::FP32, rt::Device::CPU);
    rt::ops::linear(x_cpu, w_cpu, b_cpu, y_cpu);

    rt::cuda::DeviceBuffer x_buf, w_buf, b_buf, y_buf(static_cast<size_t>(T) * out_f * sizeof(float));
    rt::Tensor x_gpu = to_device(x_h, rt::Shape{T, in_f}, x_buf);
    rt::Tensor w_gpu = to_device(w_h, rt::Shape{in_f, out_f}, w_buf);
    rt::Tensor b_gpu = to_device(b_h, rt::Shape{out_f}, b_buf);
    rt::Tensor y_gpu(y_buf.data(), rt::Shape{T, out_f}, rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::ops::linear(x_gpu, w_gpu, b_gpu, y_gpu);
    auto y_gpu_h = to_host(y_buf, T * out_f);

    float diff = max_abs_diff(y_cpu_h, y_gpu_h);
    std::printf("linear:      max_abs_diff=%.6g %s\n", diff, diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

void test_softmax(float atol) {
    const uint32_t rows = 5, cols = 11;
    auto x_h = random_vec(rows * cols, -3.0f, 3.0f);

    std::vector<float> x_cpu_h = x_h;
    rt::Tensor x_cpu(x_cpu_h.data(), rt::Shape{rows, cols}, rt::DataType::FP32, rt::Device::CPU);
    rt::ops::softmax_rows(x_cpu);

    rt::cuda::DeviceBuffer x_buf;
    rt::Tensor x_gpu = to_device(x_h, rt::Shape{rows, cols}, x_buf);
    rt::cuda::ops::softmax_rows(x_gpu);
    auto x_gpu_h = to_host(x_buf, rows * cols);

    float diff = max_abs_diff(x_cpu_h, x_gpu_h);
    std::printf("softmax_rows: max_abs_diff=%.6g %s\n", diff, diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

void test_layer_norm(float atol) {
    const uint32_t T = 6, D = 20;
    auto x_h = random_vec(T * D);
    auto w_h = random_vec(D, 0.5f, 1.5f);
    auto b_h = random_vec(D);

    rt::Tensor x_cpu(x_h.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CPU);
    rt::Tensor w_cpu(w_h.data(), rt::Shape{D}, rt::DataType::FP32, rt::Device::CPU);
    rt::Tensor b_cpu(b_h.data(), rt::Shape{D}, rt::DataType::FP32, rt::Device::CPU);
    std::vector<float> y_cpu_h(T * D);
    rt::Tensor y_cpu(y_cpu_h.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CPU);
    rt::transformer::layer_norm(x_cpu, w_cpu, b_cpu, y_cpu);

    rt::cuda::DeviceBuffer x_buf, w_buf, b_buf, y_buf(static_cast<size_t>(T) * D * sizeof(float));
    rt::Tensor x_gpu = to_device(x_h, rt::Shape{T, D}, x_buf);
    rt::Tensor w_gpu = to_device(w_h, rt::Shape{D}, w_buf);
    rt::Tensor b_gpu = to_device(b_h, rt::Shape{D}, b_buf);
    rt::Tensor y_gpu(y_buf.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::transformer::layer_norm(x_gpu, w_gpu, b_gpu, y_gpu);
    auto y_gpu_h = to_host(y_buf, T * D);

    float diff = max_abs_diff(y_cpu_h, y_gpu_h);
    std::printf("layer_norm:  max_abs_diff=%.6g %s\n", diff, diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

void test_gelu(float atol) {
    const size_t n = 100;
    auto x_h = random_vec(n, -4.0f, 4.0f);

    std::vector<float> y_cpu_h(n);
    for (size_t i = 0; i < n; ++i) y_cpu_h[i] = rt::ops::gelu(x_h[i]);

    rt::cuda::DeviceBuffer x_buf;
    rt::Tensor x_gpu = to_device(x_h, rt::Shape{static_cast<uint32_t>(n)}, x_buf);
    rt::cuda::ops::gelu_inplace(x_gpu);
    auto y_gpu_h = to_host(x_buf, n);

    float diff = max_abs_diff(y_cpu_h, y_gpu_h);
    std::printf("gelu:        max_abs_diff=%.6g %s\n", diff, diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

void test_embedding(float atol) {
    const uint32_t V = 15, D = 12, context_length = 32;
    auto tok_h = random_vec(V * D);
    auto pos_h = random_vec(context_length * D);
    std::vector<int32_t> ids = {3, 7, 1, 14, 0, 9};
    const uint32_t T = static_cast<uint32_t>(ids.size());
    const uint32_t position_offset = 5;  // exercise the Phase 2 decode-step path too

    rt::Tensor tok_cpu(tok_h.data(), rt::Shape{V, D}, rt::DataType::FP32, rt::Device::CPU);
    rt::Tensor pos_cpu(pos_h.data(), rt::Shape{context_length, D}, rt::DataType::FP32, rt::Device::CPU);
    std::vector<float> y_cpu_h(T * D);
    rt::Tensor y_cpu(y_cpu_h.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CPU);
    rt::transformer::embedding_forward(tok_cpu, pos_cpu, ids, y_cpu, position_offset);

    rt::cuda::DeviceBuffer tok_buf, pos_buf, y_buf(static_cast<size_t>(T) * D * sizeof(float));
    rt::Tensor tok_gpu = to_device(tok_h, rt::Shape{V, D}, tok_buf);
    rt::Tensor pos_gpu = to_device(pos_h, rt::Shape{context_length, D}, pos_buf);
    rt::Tensor y_gpu(y_buf.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::transformer::embedding_forward(tok_gpu, pos_gpu, ids, y_gpu, position_offset);
    auto y_gpu_h = to_host(y_buf, T * D);

    float diff = max_abs_diff(y_cpu_h, y_gpu_h);
    std::printf("embedding:   max_abs_diff=%.6g %s\n", diff, diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

void test_attention(float atol) {
    const uint32_t T = 9, D = 16, H = 4;
    auto x_h = random_vec(T * D);
    auto wq_w_h = random_vec(D * D), wq_b_h = random_vec(D);
    auto wk_w_h = random_vec(D * D), wk_b_h = random_vec(D);
    auto wv_w_h = random_vec(D * D), wv_b_h = random_vec(D);
    auto wo_w_h = random_vec(D * D), wo_b_h = random_vec(D);

    rt::transformer::AttentionWeights w_cpu;
    rt::Tensor x_cpu(x_h.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wq_w = rt::Tensor(wq_w_h.data(), rt::Shape{D, D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wq_b = rt::Tensor(wq_b_h.data(), rt::Shape{D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wk_w = rt::Tensor(wk_w_h.data(), rt::Shape{D, D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wk_b = rt::Tensor(wk_b_h.data(), rt::Shape{D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wv_w = rt::Tensor(wv_w_h.data(), rt::Shape{D, D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wv_b = rt::Tensor(wv_b_h.data(), rt::Shape{D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wo_w = rt::Tensor(wo_w_h.data(), rt::Shape{D, D}, rt::DataType::FP32, rt::Device::CPU);
    w_cpu.wo_b = rt::Tensor(wo_b_h.data(), rt::Shape{D}, rt::DataType::FP32, rt::Device::CPU);
    std::vector<float> y_cpu_h(T * D);
    rt::Tensor y_cpu(y_cpu_h.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CPU);
    rt::transformer::causal_self_attention(x_cpu, w_cpu, static_cast<int>(H), y_cpu);

    rt::cuda::DeviceBuffer x_buf, wq_w_buf, wq_b_buf, wk_w_buf, wk_b_buf, wv_w_buf, wv_b_buf,
        wo_w_buf, wo_b_buf, y_buf(static_cast<size_t>(T) * D * sizeof(float));
    rt::transformer::AttentionWeights w_gpu;
    rt::Tensor x_gpu = to_device(x_h, rt::Shape{T, D}, x_buf);
    w_gpu.wq_w = to_device(wq_w_h, rt::Shape{D, D}, wq_w_buf);
    w_gpu.wq_b = to_device(wq_b_h, rt::Shape{D}, wq_b_buf);
    w_gpu.wk_w = to_device(wk_w_h, rt::Shape{D, D}, wk_w_buf);
    w_gpu.wk_b = to_device(wk_b_h, rt::Shape{D}, wk_b_buf);
    w_gpu.wv_w = to_device(wv_w_h, rt::Shape{D, D}, wv_w_buf);
    w_gpu.wv_b = to_device(wv_b_h, rt::Shape{D}, wv_b_buf);
    w_gpu.wo_w = to_device(wo_w_h, rt::Shape{D, D}, wo_w_buf);
    w_gpu.wo_b = to_device(wo_b_h, rt::Shape{D}, wo_b_buf);
    rt::Tensor y_gpu(y_buf.data(), rt::Shape{T, D}, rt::DataType::FP32, rt::Device::CUDA);
    rt::cuda::transformer::causal_self_attention(x_gpu, w_gpu, static_cast<int>(H), y_gpu);
    auto y_gpu_h = to_host(y_buf, T * D);

    float diff = max_abs_diff(y_cpu_h, y_gpu_h);
    std::printf("attention:   max_abs_diff=%.6g %s\n", diff, diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

void test_full_model_if_available(const char* model_path, float atol) {
    rt::model::Model model;
    try {
        model = rt::model::load_model(model_path);
    } catch (const std::exception& e) {
        std::printf("full model forward: SKIPPED (%s)\n", e.what());
        return;
    }

    const std::vector<int32_t> ids = {7, 15, 12, 1, 10, 8, 27};  // "The cat" (case_0)
    rt::execution::ForwardResult cpu_result = rt::execution::forward(model, ids);

    rt::cuda::CudaModel gpu_model = rt::cuda::upload_to_cuda(model);
    rt::execution::ForwardResult gpu_result = rt::execution::forward_cuda(gpu_model, ids);

    float diff = max_abs_diff(cpu_result.logits, gpu_result.logits);
    std::printf("full model forward (logits): max_abs_diff=%.6g %s\n", diff,
                diff <= atol ? "OK" : "FAIL");
    CHECK(diff <= atol);
}

}  // namespace

int main(int argc, char** argv) {
    const float atol = 1e-3f;
    test_linear(atol);
    test_softmax(atol);
    test_layer_norm(atol);
    test_gelu(atol);
    test_embedding(atol);
    test_attention(atol);

    const char* model_path = argc > 1 ? argv[1] : "models/model.bin";
    test_full_model_if_available(model_path, atol);

    if (g_failures == 0) {
        std::printf("cuda_ops_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "cuda_ops_test: %d check(s) failed\n", g_failures);
    return 1;
}
