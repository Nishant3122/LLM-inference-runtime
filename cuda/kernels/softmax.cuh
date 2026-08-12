// Naive row-wise softmax, in place. Mirrors runtime/transformer/ops.h::softmax_rows.
#pragma once

#include "../../runtime/core/tensor.h"

namespace rt::cuda::ops {

// x must be a device tensor, shape [rows, cols]. One thread per row (not one block
// per row with a parallel reduction — that's the Phase 4 version); each thread scans
// its row twice (max, then exp+sum) sequentially, same op order as the CPU version.
// Fine at this model's cols size (<= context_length, 256 for Stage 1); revisit if a
// later stage's cols grows enough that per-thread sequential scanning becomes the
// bottleneck profiling flags.
void softmax_rows(rt::Tensor& x);

}  // namespace rt::cuda::ops
