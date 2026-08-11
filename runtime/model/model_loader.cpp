#include "model_loader.h"

#include <cstring>
#include <fstream>
#include <sstream>

#include "../core/shape.h"
#include "../core/types.h"

namespace rt::model {

namespace {

constexpr uint32_t kMagic = 0x544C4D52;  // "TLMR" little-endian
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kDtypeFp32 = 0;
constexpr size_t kHeaderSize = 64;
constexpr size_t kEntryNameSize = 64;
constexpr size_t kEntrySize = 104;

#pragma pack(push, 1)
struct RawHeader {
    uint32_t magic;
    uint32_t format_version;
    uint32_t architecture;
    uint32_t vocab_size;
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t d_ff;
    uint32_t context_length;
    uint32_t tensor_count;
    uint32_t tensor_table_offset;
    uint32_t data_section_offset;
    uint8_t reserved[16];
};

struct RawTensorEntry {
    char name[kEntryNameSize];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t nbytes;
};
#pragma pack(pop)

static_assert(sizeof(RawHeader) == kHeaderSize, "RawHeader layout must match docs/model_format.md");
static_assert(sizeof(RawTensorEntry) == kEntrySize, "RawTensorEntry layout must match docs/model_format.md");

std::string entry_name_to_string(const char* name) {
    // NUL-padded, but not guaranteed NUL-terminated if all 64 bytes are used.
    size_t len = 0;
    while (len < kEntryNameSize && name[len] != '\0') ++len;
    return std::string(name, len);
}

const Tensor& require(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("model.bin: missing required tensor '" + name + "'");
    }
    return it->second;
}

}  // namespace

Model load_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("model.bin: failed to open " + path);
    }

    RawHeader header{};
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!f) {
        throw std::runtime_error("model.bin: file too small for header (" + path + ")");
    }
    if (header.magic != kMagic) {
        std::ostringstream oss;
        oss << "model.bin: bad magic 0x" << std::hex << header.magic
            << " (expected 0x" << kMagic << ")";
        throw std::runtime_error(oss.str());
    }
    if (header.format_version != kFormatVersion) {
        throw std::runtime_error("model.bin: unsupported format_version " +
                                  std::to_string(header.format_version));
    }

    Model model;
    model.config.architecture = static_cast<Architecture>(header.architecture);
    model.config.vocab_size = header.vocab_size;
    model.config.d_model = header.d_model;
    model.config.n_layers = header.n_layers;
    model.config.n_heads = header.n_heads;
    model.config.d_ff = header.d_ff;
    model.config.context_length = header.context_length;

    // --- tensor table ---
    f.seekg(header.tensor_table_offset, std::ios::beg);
    std::vector<RawTensorEntry> entries(header.tensor_count);
    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        f.read(reinterpret_cast<char*>(&entries[i]), sizeof(RawTensorEntry));
        if (!f) {
            throw std::runtime_error("model.bin: unexpected EOF reading tensor table entry " +
                                      std::to_string(i));
        }
    }

    // --- data section: read it all into one owned buffer; tensors are views into it ---
    f.seekg(0, std::ios::end);
    std::streamoff file_size = f.tellg();
    if (file_size < static_cast<std::streamoff>(header.data_section_offset)) {
        throw std::runtime_error("model.bin: file smaller than its declared data_section_offset");
    }
    size_t data_size = static_cast<size_t>(file_size) - header.data_section_offset;
    model.storage.resize(data_size);
    f.seekg(header.data_section_offset, std::ios::beg);
    if (data_size > 0) {
        f.read(reinterpret_cast<char*>(model.storage.data()), static_cast<std::streamsize>(data_size));
        if (!f) {
            throw std::runtime_error("model.bin: unexpected EOF reading tensor data section");
        }
    }

    for (const auto& e : entries) {
        if (e.dtype != kDtypeFp32) {
            throw std::runtime_error(
                "model.bin: tensor '" + entry_name_to_string(e.name) +
                "' has unsupported dtype " + std::to_string(e.dtype) +
                " (only FP32 is supported until Phase 5 quantization)");
        }
        if (e.offset + e.nbytes > model.storage.size()) {
            throw std::runtime_error("model.bin: tensor '" + entry_name_to_string(e.name) +
                                      "' data range exceeds file bounds");
        }
        Shape shape;
        shape.ndim = static_cast<int>(e.ndim);
        for (int d = 0; d < shape.ndim; ++d) shape.dims[d] = e.shape[d];

        Tensor t(model.storage.data() + e.offset, shape, DataType::FP32, Device::CPU);
        model.tensors_by_name.emplace(entry_name_to_string(e.name), t);
    }

    // --- populate named fields from the map ---
    const auto& t = model.tensors_by_name;
    model.tok_embedding = require(t, "tok_embedding.weight");
    model.pos_embedding = require(t, "pos_embedding.weight");

    model.layers.resize(model.config.n_layers);
    for (uint32_t i = 0; i < model.config.n_layers; ++i) {
        std::string p = "layers." + std::to_string(i) + ".";
        auto& layer = model.layers[i];
        layer.ln1_w = require(t, p + "ln1.weight");
        layer.ln1_b = require(t, p + "ln1.bias");
        layer.attn.wq_w = require(t, p + "attn.wq.weight");
        layer.attn.wq_b = require(t, p + "attn.wq.bias");
        layer.attn.wk_w = require(t, p + "attn.wk.weight");
        layer.attn.wk_b = require(t, p + "attn.wk.bias");
        layer.attn.wv_w = require(t, p + "attn.wv.weight");
        layer.attn.wv_b = require(t, p + "attn.wv.bias");
        layer.attn.wo_w = require(t, p + "attn.wo.weight");
        layer.attn.wo_b = require(t, p + "attn.wo.bias");
        layer.ln2_w = require(t, p + "ln2.weight");
        layer.ln2_b = require(t, p + "ln2.bias");
        layer.mlp.fc1_w = require(t, p + "mlp.fc1.weight");
        layer.mlp.fc1_b = require(t, p + "mlp.fc1.bias");
        layer.mlp.fc2_w = require(t, p + "mlp.fc2.weight");
        layer.mlp.fc2_b = require(t, p + "mlp.fc2.bias");
    }

    model.ln_f_w = require(t, "ln_f.weight");
    model.ln_f_b = require(t, "ln_f.bias");
    model.lm_head_w = require(t, "lm_head.weight");
    model.lm_head_b = require(t, "lm_head.bias");

    return model;
}

}  // namespace rt::model
