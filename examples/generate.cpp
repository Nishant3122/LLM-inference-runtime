// Phase 1 CPU generation, mirroring reference/generate.py: load model.bin + vocab.bin
// through the custom runtime (no PyTorch/libtorch involved — G2, docs/architecture.md
// §4), encode a prompt, autoregressively sample tokens, decode, print.
//
// No KV cache yet (that's Phase 2): every step recomputes the full forward pass over
// the whole sequence so far. This is intentionally the naive baseline Phase 2's
// with/without-cache ablation (spec §32) will be measured against.
//
// Usage:
//   generate_example [model.bin] [vocab.bin] [prompt] [max_new_tokens] [temperature] [top_k]
// All arguments optional; defaults mirror reference/generate.py's.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../runtime/core/version.h"
#include "../runtime/execution/cpu_backend.h"
#include "../runtime/model/model_loader.h"
#include "../runtime/sampling/greedy.h"
#include "../runtime/sampling/temperature.h"
#include "../runtime/sampling/top_k.h"
#include "../tools/tokenizer/tokenizer.h"

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "models/model.bin";
    std::string vocab_path = argc > 2 ? argv[2] : "models/vocab.bin";
    std::string prompt = argc > 3 ? argv[3] : "The cat";
    int max_new_tokens = argc > 4 ? std::atoi(argv[4]) : 200;
    float temperature = argc > 5 ? static_cast<float>(std::atof(argv[5])) : 0.8f;
    int top_k = argc > 6 ? std::atoi(argv[6]) : 20;

    std::printf("adaptive-llm-runtime %s (Phase 1: CPU, no KV cache)\n", rt::runtime_version());

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
    const uint32_t context_length = model.config.context_length;

    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<int32_t> ids_cond = ids;
        if (ids_cond.size() > context_length) {
            ids_cond.assign(ids.end() - context_length, ids.end());
        }

        rt::execution::ForwardResult result = rt::execution::forward(model, ids_cond);
        const float* last_logits =
            result.logits.data() + static_cast<size_t>(result.T - 1) * result.V;

        int32_t next_id;
        if (temperature <= 0.0f) {
            next_id = rt::sampling::greedy_sample(last_logits, static_cast<int>(result.V));
        } else if (top_k > 0) {
            next_id = rt::sampling::top_k_sample(last_logits, static_cast<int>(result.V), top_k,
                                                  temperature, rng);
        } else {
            next_id = rt::sampling::temperature_sample(last_logits, static_cast<int>(result.V),
                                                         temperature, rng);
        }
        ids.push_back(next_id);
    }

    std::string output = tokenizer.decode(ids);
    std::printf("%s\n", output.c_str());
    return 0;
}
