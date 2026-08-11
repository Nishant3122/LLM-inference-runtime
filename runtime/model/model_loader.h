// Reads model.bin (docs/model_format.md: header + tensor table + data section) into
// a Model. The one place in the runtime that parses that byte layout — keep it in
// sync with reference/export.py, the Python side that writes it.
#pragma once

#include <string>

#include "model.h"

namespace rt::model {

// Throws std::runtime_error on: missing file, bad magic, unsupported format_version,
// an unsupported dtype (only FP32 is supported until Phase 5 quantization), or a
// missing required tensor name.
Model load_model(const std::string& path);

}  // namespace rt::model
