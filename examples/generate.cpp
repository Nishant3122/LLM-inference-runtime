// Phase 2 CPU generation: prefill the prompt once, then decode one token at a time
// using a persistent KV cache — no more recomputing the whole sequence from scratch
// every step (that was Phase 1's baseline). Pass --no-cache to fall back to the
// Phase 1 path (repeated full forward() calls) so both are runnable from the same
// binary for the with/without-cache benchmark (spec §32 ablation, Q1).
//
// Usage:
//   generate_example [model.bin] [vocab.bin] [prompt] [max_new_tokens] [temperature] [top_k] [--no-cache]
// All positional arguments optional; defaults mirror reference/generate.py's.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../runtime/cache/cache_manager.h"
#include "../runtime/core/version.h"
#include "../runtime/execution/cpu_backend.h"
#include "../runtime/model/model_loader.h"
#include "../runtime/sampling/greedy.h"
#include "../runtime/sampling/temperature.h"
#include "../runtime/sampling/top_k.h"
#include "../tools/tokenizer/tokenizer.h"

namespace {

int32_t sample_next(const float* logits, int vocab_size, float temperature, int top_k,
                     std::mt19937& rng) {
    if (temperature <= 0.0f) return rt::sampling::greedy_sample(logits, vocab_size);
    if (top_k > 0) return rt::sampling::top_k_sample(logits, vocab_size, top_k, temperature, rng);
    return rt::sampling::temperature_sample(logits, vocab_size, temperature, rng);
}

// Phase 1 baseline: recompute the full forward pass (over a sliding window of at
// most context_length tokens) on every step. Kept for the with/without-cache
// benchmark; see runtime/execution/README.md for measured timings.
std::vector<int32_t> generate_no_cache(const rt::model::Model& model, std::vector<int32_t> ids,
                                        int max_new_tokens, float temperature, int top_k,
                                        std::mt19937& rng) {
    const uint32_t context_length = model.config.context_length;
    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<int32_t> ids_cond = ids;
        if (ids_cond.size() > context_length) {
            ids_cond.assign(ids.end() - context_length, ids.end());
        }
        rt::execution::ForwardResult result = rt::execution::forward(model, ids_cond);
        const float* last_logits =
            result.logits.data() + static_cast<size_t>(result.T - 1) * result.V;
        ids.push_back(sample_next(last_logits, static_cast<int>(result.V), temperature, top_k, rng));
    }
    return ids;
}

// Phase 2: prefill once, then one decode_step() per new token — each step only does
// O(1) new work per layer (one token's Q/K/V + attending over the cache) instead of
// recomputing the whole sequence.
std::vector<int32_t> generate_cached(const rt::model::Model& model, std::vector<int32_t> ids,
                                      int max_new_tokens, float temperature, int top_k,
                                      std::mt19937& rng) {
    rt::cache::KVCache kv = rt::cache::create_cache(model.config.n_layers,
                                                     model.config.context_length,
                                                     model.config.d_model);

    rt::execution::ForwardResult prefill_result = rt::execution::prefill(model, ids, kv);
    const float* last_logits = prefill_result.logits.data() +
                                static_cast<size_t>(prefill_result.T - 1) * prefill_result.V;
    int32_t next_id =
        sample_next(last_logits, static_cast<int>(prefill_result.V), temperature, top_k, rng);
    ids.push_back(next_id);

    for (int step = 1; step < max_new_tokens; ++step) {
        if (kv.length() >= model.config.context_length) {
            std::fprintf(stderr,
                         "note: context_length (%u) reached, stopping early "
                         "(Phase 2 has no sliding-window eviction yet)\n",
                         model.config.context_length);
            break;
        }
        uint32_t position = kv.length();  // absolute position of next_id, about to be fed in
        std::vector<float> logits = rt::execution::decode_step(model, next_id, position, kv);
        next_id = sample_next(logits.data(), static_cast<int>(logits.size()), temperature, top_k, rng);
        ids.push_back(next_id);
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    bool use_cache = true;
    {
        auto it = std::find(args.begin(), args.end(), "--no-cache");
        if (it != args.end()) {
            use_cache = false;
            args.erase(it);
        }
    }

    std::string model_path = args.size() > 0 ? args[0] : "models/model.bin";
    std::string vocab_path = args.size() > 1 ? args[1] : "models/vocab.bin";
    std::string prompt = args.size() > 2 ? args[2] : "The cat";
    int max_new_tokens = args.size() > 3 ? std::atoi(args[3].c_str()) : 200;
    float temperature = args.size() > 4 ? static_cast<float>(std::atof(args[4].c_str())) : 0.8f;
    int top_k = args.size() > 5 ? std::atoi(args[5].c_str()) : 20;

    std::printf("adaptive-llm-runtime %s (Phase 2: CPU, %s)\n", rt::runtime_version(),
                use_cache ? "KV cache" : "no cache, --no-cache");

    rt::model::Model model;
    tok::CharTokenizer tokenizer;
    try {
        model = rt::model::load_model(model_path);
        tokenizer = tok::CharTokenizer::load(vocab_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "(expected model.bin/vocab.bin from `python reference/export.py`)\n");
        return 1;
    }

    std::printf("Model: vocab=%u d_model=%u n_layers=%u n_heads=%u d_ff=%u context_length=%u\n",
                model.config.vocab_size, model.config.d_model, model.config.n_layers,
                model.config.n_heads, model.config.d_ff, model.config.context_length);
    std::printf("Prompt: %s\n\n", prompt.c_str());

    std::vector<int32_t> ids;
    try {
        ids = tokenizer.encode(prompt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error encoding prompt: %s\n", e.what());
        return 1;
    }

    std::mt19937 rng(std::random_device{}());

    auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> out_ids =
        use_cache ? generate_cached(model, ids, max_new_tokens, temperature, top_k, rng)
                  : generate_no_cache(model, ids, max_new_tokens, temperature, top_k, rng);
    auto t1 = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    std::string output = tokenizer.decode(out_ids);
    std::printf("%s\n\n", output.c_str());
    std::printf("[%d new tokens in %.3fs = %.1f tok/s]\n",
                static_cast<int>(out_ids.size() - ids.size()), seconds,
                (out_ids.size() - ids.size()) / seconds);
    return 0;
}
