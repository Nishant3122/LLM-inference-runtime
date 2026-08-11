// Correctness test for Phase 2's KV cache: feeds the same fixed token sequence
// through both the Phase 1 "recompute everything" path (execution::forward) and the
// Phase 2 "prefill + incremental decode" path (execution::prefill +
// execution::decode_step), and checks every step's logits match. This is the
// with/without-cache ablation from spec §32, phrased as a correctness check rather
// than a benchmark (the benchmark is examples/generate.cpp --no-cache timing, see
// runtime/execution/README.md).
//
// Expected to match near-bit-exactly, not just "close": every op here (LayerNorm,
// linear, attention weighted-sum) operates on one query row independently of any
// other row being computed alongside it, and the cached path's summation order for a
// given row is identical to the uncached path's (see the comment on
// causal_self_attention_cached in runtime/transformer/attention.h). A large diff here
// would mean the two code paths are doing different math, not just accumulating FP
// noise differently — that's a real bug, not a tolerance issue.
//
// Input ids copied from tests/golden/manifest.json case_1 (see tests/model_test.cpp
// for the same convention).
//
// Usage: kv_cache_test <path to model.bin>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../runtime/execution/cpu_backend.h"
#include "../runtime/model/model_loader.h"

namespace {
int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 2;
    }

    rt::model::Model model;
    try {
        model = rt::model::load_model(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load model: %s\n", e.what());
        return 2;
    }

    // tests/golden/manifest.json, case_1: "A small bird finds"
    const std::vector<int32_t> full_ids = {4,  1,  26, 20, 8,  19, 19, 1, 9,
                                            16, 25, 11, 1,  13, 16, 21, 11, 26};
    const size_t prefill_len = 7;  // arbitrary split: prefill 7, decode_step the rest
    const float atol = 1e-4f;

    rt::cache::KVCache kv;
    kv.init(model.config.n_layers, model.config.context_length, model.config.d_model);

    std::vector<int32_t> prefill_ids(full_ids.begin(), full_ids.begin() + prefill_len);
    rt::execution::ForwardResult prefill_result = rt::execution::prefill(model, prefill_ids, kv);
    CHECK(kv.length() == prefill_len);

    // prefill() must be numerically identical to forward() on the same ids (same
    // computation; kv-seeding is a side effect that doesn't touch the returned values).
    {
        rt::execution::ForwardResult plain = rt::execution::forward(model, prefill_ids);
        float diff = max_abs_diff(prefill_result.logits, plain.logits);
        std::printf("prefill vs forward (ids[0:%zu]): max_abs_diff=%.6g %s\n", prefill_len, diff,
                    diff <= atol ? "OK" : "FAIL");
        CHECK(diff <= atol);
    }

    // Now decode_step through the rest, comparing each step's logits against a
    // from-scratch forward() over the sequence seen so far.
    for (size_t t = prefill_len; t < full_ids.size(); ++t) {
        uint32_t position = kv.length();
        CHECK(position == static_cast<uint32_t>(t));

        std::vector<float> decoded_logits = rt::execution::decode_step(model, full_ids[t], position, kv);

        std::vector<int32_t> ids_so_far(full_ids.begin(), full_ids.begin() + t + 1);
        rt::execution::ForwardResult reference = rt::execution::forward(model, ids_so_far);
        std::vector<float> reference_last_row(
            reference.logits.end() - reference.V, reference.logits.end());

        float diff = max_abs_diff(decoded_logits, reference_last_row);
        std::printf("decode_step t=%2zu vs forward(ids[0:%2zu]): max_abs_diff=%.6g %s\n", t, t + 1,
                    diff, diff <= atol ? "OK" : "FAIL");
        CHECK(diff <= atol);
        CHECK(kv.length() == t + 1);
    }

    if (g_failures == 0) {
        std::printf("kv_cache_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "kv_cache_test: %d check(s) failed\n", g_failures);
    return 1;
}
