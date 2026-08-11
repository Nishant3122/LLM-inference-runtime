// Correctness test for the full Phase 1 CPU forward pass: load model.bin, run the
// two golden prompts, compare every intermediate activation (embedding, each block,
// final norm, logits) against reference/dump_golden.py's PyTorch-produced golden
// values. This is G1 (docs/architecture.md §4) — the actual milestone Phase 1 exists
// to hit.
//
// Input ids are copied from tests/golden/manifest.json rather than parsed at runtime
// (no JSON parser in C++ — Engineering Principle 4); regenerate these constants if
// the model is ever retrained with a different vocab/corpus.
//
// Usage: model_test <path to model.bin> <path to tests/golden dir>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../runtime/execution/cpu_backend.h"
#include "../runtime/model/model_loader.h"

namespace {
int g_failures = 0;

struct Golden2D {
    uint32_t rows = 0, cols = 0;
    std::vector<float> data;
};

Golden2D load_golden(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open golden file: " + path);
    Golden2D g;
    f.read(reinterpret_cast<char*>(&g.rows), sizeof(g.rows));
    f.read(reinterpret_cast<char*>(&g.cols), sizeof(g.cols));
    g.data.resize(static_cast<size_t>(g.rows) * g.cols);
    f.read(reinterpret_cast<char*>(g.data.data()),
           static_cast<std::streamsize>(g.data.size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated golden file: " + path);
    return g;
}

// Elementwise max-abs-diff check with a small fixed tolerance: both sides are FP32
// doing the same math with different summation order (naive C++ loops vs. PyTorch's
// BLAS), so exact equality isn't expected — see tests/README.md "Numerical tolerance".
bool check_close(const std::string& label, const std::vector<float>& actual,
                  const std::vector<float>& expected, float atol) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "FAIL %s: size mismatch (actual=%zu expected=%zu)\n",
                     label.c_str(), actual.size(), expected.size());
        ++g_failures;
        return false;
    }
    float max_abs_diff = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }
    bool ok = max_abs_diff <= atol;
    std::printf("%-28s max_abs_diff=%.6g (at idx %zu: actual=%.6g expected=%.6g) %s\n",
                label.c_str(), max_abs_diff, max_idx, actual[max_idx], expected[max_idx],
                ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
    return ok;
}

void run_case(const rt::model::Model& model, const std::string& golden_dir,
              const std::string& case_name, const std::vector<int32_t>& ids,
              int32_t expected_greedy_next_id, float atol) {
    std::printf("\n-- %s (T=%zu) --\n", case_name.c_str(), ids.size());
    rt::execution::ForwardResult result = rt::execution::forward(model, ids);

    std::string dir = golden_dir + "/" + case_name + "/";
    check_close(case_name + "/embedding_output", result.embedding_output,
                load_golden(dir + "embedding_output.bin").data, atol);

    for (size_t i = 0; i < result.block_outputs.size(); ++i) {
        std::string basename = "block_" + std::to_string(i) + "_output";
        check_close(case_name + "/" + basename, result.block_outputs[i],
                    load_golden(dir + basename + ".bin").data, atol);
    }

    check_close(case_name + "/final_norm_output", result.final_norm_output,
                load_golden(dir + "final_norm_output.bin").data, atol);
    check_close(case_name + "/logits", result.logits, load_golden(dir + "logits.bin").data, atol);

    // greedy argmax of the last position's logits should match the golden model's,
    // independent of the raw float tolerance above (catches "close but wrong token").
    const float* last = result.logits.data() + static_cast<size_t>(result.T - 1) * result.V;
    int32_t best = 0;
    for (uint32_t v = 1; v < result.V; ++v) {
        if (last[v] > last[best]) best = static_cast<int32_t>(v);
    }
    bool ok = (best == expected_greedy_next_id);
    std::printf("%-28s actual=%d expected=%d %s\n", (case_name + "/greedy_next_id").c_str(), best,
                expected_greedy_next_id, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.bin> <tests/golden dir>\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    std::string golden_dir = argv[2];

    rt::model::Model model;
    try {
        model = rt::model::load_model(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load model: %s\n", e.what());
        return 2;
    }
    std::printf("Loaded model: vocab=%u d_model=%u n_layers=%u n_heads=%u\n",
                model.config.vocab_size, model.config.d_model, model.config.n_layers,
                model.config.n_heads);

    // Observed max_abs_diff against the golden PyTorch outputs is ~1e-6 (pure FP32
    // rounding-order noise between this naive triple-loop matmul and PyTorch's BLAS
    // path) — 1e-3 leaves ~1000x margin while still catching a real correctness bug.
    // See tests/README.md "Numerical tolerance".
    const float atol = 1e-3f;

    try {
        // tests/golden/manifest.json, case_0: "The cat"
        run_case(model, golden_dir, "case_0", {7, 15, 12, 1, 10, 8, 27}, 1, atol);

        // tests/golden/manifest.json, case_1: "A small bird finds"
        run_case(model, golden_dir, "case_1",
                 {4, 1, 26, 20, 8, 19, 19, 1, 9, 16, 25, 11, 1, 13, 16, 21, 11, 26}, 1, atol);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    if (g_failures == 0) {
        std::printf("\nmodel_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "\nmodel_test: %d check(s) failed\n", g_failures);
    return 1;
}
