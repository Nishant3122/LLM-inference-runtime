// Unit test for runtime/core: Shape and Tensor.
//
// No external test framework: fetching one (e.g. GoogleTest via FetchContent) would
// need a network download this project doesn't want as a hard build dependency for
// Phase 0/1. Plain asserts + a manual pass/fail summary are enough for the handful of
// core-type tests Phase 1 needs; revisit if the test surface grows past what this
// pattern can hold cleanly.
//
// Build-verified 2026-08-12 with CMake 4.4.2 + MSVC 19.44:
//   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
//   cmake --build build --config RelWithDebInfo --target tensor_test
//   ctest --test-dir build -C RelWithDebInfo -R tensor_test

#include <cassert>
#include <cstdio>
#include <vector>

#include "../runtime/core/shape.h"
#include "../runtime/core/tensor.h"
#include "../runtime/core/types.h"

using namespace rt;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

void test_shape_numel() {
    Shape s{2, 3, 4};
    CHECK(s.ndim == 3);
    CHECK(s.numel() == 24);
    CHECK(s.to_string() == "[2, 3, 4]");
}

void test_shape_equality() {
    Shape a{2, 3};
    Shape b{2, 3};
    Shape c{3, 2};
    CHECK(a == b);
    CHECK(a != c);
}

void test_shape_empty() {
    Shape s;
    CHECK(s.ndim == 0);
    CHECK(s.numel() == 0);
}

void test_dtype_size() {
    CHECK(dtype_size(DataType::FP32) == 4);
    CHECK(dtype_size(DataType::FP16) == 2);
    CHECK(dtype_size(DataType::INT8) == 1);
}

void test_tensor_nbytes() {
    std::vector<float> buf(2 * 3 * 4, 0.0f);
    Tensor t(buf.data(), Shape{2, 3, 4}, DataType::FP32, Device::CPU);
    CHECK(t.numel() == 24);
    CHECK(t.nbytes() == 24 * sizeof(float));
    CHECK(t.is_cpu());
    CHECK(!t.is_cuda());
}

}  // namespace

int main() {
    test_shape_numel();
    test_shape_equality();
    test_shape_empty();
    test_dtype_size();
    test_tensor_nbytes();

    if (g_failures == 0) {
        std::printf("tensor_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "tensor_test: %d check(s) failed\n", g_failures);
    return 1;
}
