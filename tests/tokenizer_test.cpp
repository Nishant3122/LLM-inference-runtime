// Unit test for tools/tokenizer against tests/golden/manifest.json's known
// prompt -> id mappings (values copied here as constants; regenerate by re-reading
// manifest.json if the training corpus/model is ever retrained with a different
// vocab — see tests/README.md).
//
// Usage: tokenizer_test <path to vocab.bin>
#include <cstdio>
#include <vector>

#include "../tools/tokenizer/tokenizer.h"

namespace {
int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <vocab.bin>\n", argv[0]);
        return 2;
    }

    tok::CharTokenizer tokenizer;
    try {
        tokenizer = tok::CharTokenizer::load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load %s: %s\n", argv[1], e.what());
        return 2;
    }

    CHECK(tokenizer.vocab_size() == 33);

    // From tests/golden/manifest.json, case_0.
    {
        std::vector<int32_t> expected = {7, 15, 12, 1, 10, 8, 27};
        std::vector<int32_t> actual = tokenizer.encode("The cat");
        CHECK(actual == expected);
        CHECK(tokenizer.decode(actual) == "The cat");
    }

    // From tests/golden/manifest.json, case_1.
    {
        std::vector<int32_t> expected = {4, 1, 26, 20, 8, 19, 19, 1, 9, 16, 25, 11,
                                          1, 13, 16, 21, 11, 26};
        std::vector<int32_t> actual = tokenizer.encode("A small bird finds");
        CHECK(actual == expected);
        CHECK(tokenizer.decode(actual) == "A small bird finds");
    }

    // Unknown character should throw, not silently drop/substitute.
    {
        bool threw = false;
        try {
            tokenizer.encode("\x01");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    if (g_failures == 0) {
        std::printf("tokenizer_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "tokenizer_test: %d check(s) failed\n", g_failures);
    return 1;
}
