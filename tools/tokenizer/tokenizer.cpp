#include "tokenizer.h"

#include <algorithm>
#include <fstream>

namespace tok {

namespace {
constexpr uint32_t kMagic = 0x54564F43;  // "TVOC" little-endian

uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("vocab.bin: unexpected EOF reading uint32");
    return v;
}
}  // namespace

CharTokenizer CharTokenizer::load(const std::string& vocab_bin_path) {
    std::ifstream f(vocab_bin_path, std::ios::binary);
    if (!f) throw std::runtime_error("vocab.bin: failed to open " + vocab_bin_path);

    uint32_t magic = read_u32(f);
    if (magic != kMagic) {
        throw std::runtime_error("vocab.bin: bad magic (expected 0x54564F43 'TVOC')");
    }
    uint32_t version = read_u32(f);
    if (version != 1) {
        throw std::runtime_error("vocab.bin: unsupported version " + std::to_string(version));
    }
    uint32_t vocab_size = read_u32(f);

    CharTokenizer tokenizer;
    tokenizer.id_to_str_.reserve(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t len = read_u32(f);
        std::string s(len, '\0');
        if (len > 0) {
            f.read(s.data(), len);
            if (!f) throw std::runtime_error("vocab.bin: unexpected EOF reading char bytes");
        }
        tokenizer.max_entry_len_ = std::max(tokenizer.max_entry_len_, s.size());
        tokenizer.str_to_id_.emplace(s, static_cast<int32_t>(i));
        tokenizer.id_to_str_.push_back(std::move(s));
    }
    return tokenizer;
}

std::vector<int32_t> CharTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        // Greedy longest-match: try the longest possible vocab entry first so
        // multi-byte UTF-8 entries (if any) win over any single-byte prefix match.
        size_t max_len = std::min(max_entry_len_, text.size() - pos);
        for (size_t len = max_len; len >= 1; --len) {
            auto it = str_to_id_.find(text.substr(pos, len));
            if (it != str_to_id_.end()) {
                ids.push_back(it->second);
                pos += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            throw std::runtime_error(
                "CharTokenizer::encode: character at byte offset " + std::to_string(pos) +
                " not in vocab");
        }
    }
    return ids;
}

std::string CharTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_str_.size()) {
            throw std::runtime_error("CharTokenizer::decode: id " + std::to_string(id) +
                                      " out of range");
        }
        out += id_to_str_[id];
    }
    return out;
}

}  // namespace tok
