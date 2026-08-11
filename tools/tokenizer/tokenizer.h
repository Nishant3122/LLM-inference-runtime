// C++ port of reference/tokenizer.py::CharTokenizer. Loads the vocab.bin format
// documented in docs/model_format.md ("Companion files: vocab.json and vocab.bin").
//
// Deliberately not a general Unicode-aware tokenizer: each vocab entry is whatever
// UTF-8 byte-string reference/tokenizer.py emitted (usually a single ASCII char for
// the Stage-1 synthetic corpus). encode() greedily matches the longest known entry
// at each position so multi-byte UTF-8 characters still round-trip correctly.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tok {

class CharTokenizer {
public:
    // Throws std::runtime_error on a missing file, bad magic, or unsupported version.
    static CharTokenizer load(const std::string& vocab_bin_path);

    int vocab_size() const { return static_cast<int>(id_to_str_.size()); }

    // Throws std::runtime_error if `text` contains a substring not in the vocab
    // (mirrors reference/tokenizer.py::CharTokenizer.encode raising ValueError).
    std::vector<int32_t> encode(const std::string& text) const;

    std::string decode(const std::vector<int32_t>& ids) const;

private:
    std::vector<std::string> id_to_str_;
    std::unordered_map<std::string, int32_t> str_to_id_;
    size_t max_entry_len_ = 1;  // longest vocab entry, in bytes; bounds the greedy match scan
};

}  // namespace tok
