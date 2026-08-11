# Model Format — `model.bin`

Custom binary format for exporting a trained `reference/model.py::TinyTransformer` for
loading by the C++ runtime, independent of PyTorch/libtorch. All integers are
little-endian. All tensor data is FP32 for Phase 0 (quantized variants come in Phase 5
and get their own `dtype` value per tensor — the format already carries a per-tensor
dtype field so mixed-precision export needs no format change).

Produced by [`reference/export.py`](../reference/export.py); the byte layout below
must stay in sync with that script and with the eventual C++ loader in
`runtime/model/model_loader`.

## Layout

```text
model.bin
├── Header               (fixed size, 64 bytes)
├── Tensor table         (variable size: tensor_count * entry)
└── Tensor data section  (raw bytes, one contiguous region per tensor)
```

### Header (64 bytes)

| Offset | Size | Field | Type | Notes |
|---|---|---|---|---|
| 0 | 4 | `magic` | uint32 | `0x544C4D52` ("TLMR" = Tiny LLM Model Runtime, ASCII bytes T,L,M,R read little-endian) |
| 4 | 4 | `format_version` | uint32 | `1` |
| 8 | 4 | `architecture` | uint32 | `0` = `tiny_transformer_decoder` (pre-norm, learned pos. embedding, tied-free LM head) |
| 12 | 4 | `vocab_size` | uint32 | |
| 16 | 4 | `d_model` | uint32 | |
| 20 | 4 | `n_layers` | uint32 | |
| 24 | 4 | `n_heads` | uint32 | must divide `d_model` |
| 28 | 4 | `d_ff` | uint32 | MLP hidden dim |
| 32 | 4 | `context_length` | uint32 | max sequence length the positional embedding supports |
| 36 | 4 | `tensor_count` | uint32 | number of entries in the tensor table |
| 40 | 4 | `tensor_table_offset` | uint32 | byte offset from file start to the tensor table (= 64) |
| 44 | 4 | `data_section_offset` | uint32 | byte offset from file start to the tensor data section |
| 48 | 16 | `reserved` | bytes | zero-filled, reserved for future header fields (e.g. quantization metadata) |

### Tensor table entry (repeated `tensor_count` times)

| Size | Field | Type | Notes |
|---|---|---|---|
| 64 | `name` | char[64] | UTF-8, NUL-padded; see naming convention below |
| 4 | `dtype` | uint32 | `0`=FP32, `1`=FP16, `2`=INT8, `3`=INT4 (Phase 0: always `0`) |
| 4 | `ndim` | uint32 | number of dimensions, ≤ 4 |
| 16 | `shape` | uint32[4] | dims, row-major order; unused trailing dims are `0` |
| 8 | `offset` | uint64 | byte offset from `data_section_offset` to this tensor's data |
| 8 | `nbytes` | uint64 | size of this tensor's data in bytes |

Entry size: 104 bytes.

### Tensor data section

Each tensor's raw data is stored contiguously, row-major (C order), starting at
`data_section_offset + entry.offset`. FP32 data is plain IEEE-754 `float32`, native
little-endian layout (matches `numpy`/`torch` default on x86/x64).

## Tensor naming convention and shapes

For a model with `L = n_layers`, `d = d_model`, `h = n_heads`, `f = d_ff`, `V = vocab_size`, `T = context_length`:

| Name | Shape | Notes |
|---|---|---|
| `tok_embedding.weight` | `[V, d]` | token embedding table |
| `pos_embedding.weight` | `[T, d]` | learned absolute positional embedding |
| `layers.{i}.ln1.weight` / `.bias` | `[d]` | pre-attention LayerNorm, `i` in `[0, L)` |
| `layers.{i}.attn.wq.weight` / `.bias` | `[d, d]` / `[d]` | |
| `layers.{i}.attn.wk.weight` / `.bias` | `[d, d]` / `[d]` | |
| `layers.{i}.attn.wv.weight` / `.bias` | `[d, d]` / `[d]` | |
| `layers.{i}.attn.wo.weight` / `.bias` | `[d, d]` / `[d]` | output projection |
| `layers.{i}.ln2.weight` / `.bias` | `[d]` | pre-MLP LayerNorm |
| `layers.{i}.mlp.fc1.weight` / `.bias` | `[d, f]` / `[f]` | GELU MLP, up-projection |
| `layers.{i}.mlp.fc2.weight` / `.bias` | `[f, d]` / `[d]` | down-projection |
| `ln_f.weight` / `.bias` | `[d]` | final LayerNorm |
| `lm_head.weight` / `.bias` | `[d, V]` / `[V]` | not weight-tied to `tok_embedding` (see architecture.md §5) |

Linear layer weights are stored as `[in_features, out_features]` (i.e. already
transposed from PyTorch's default `nn.Linear.weight` shape of `[out, in]`) so that the
C++/CUDA runtime can compute `Y = X @ W + b` directly without a transpose at load time.
`reference/export.py` performs this transpose during export.

Total tensor count: `2 + L * 16 + 4` (2 embeddings + 16 tensors/layer [ln1 w/b, wq w/b,
wk w/b, wv w/b, wo w/b, ln2 w/b, fc1 w/b, fc2 w/b] + 4 for ln_f + lm_head).
For the Stage 1 config (`L=4`): `2 + 64 + 4 = 70` tensors — confirmed against
`reference/export.py` output on the trained Stage-1 checkpoint (2026-08-12).

## Companion file: `vocab.json`

Not part of `model.bin` (tokenization is a separate concern from model weights, per
the runtime's directory split — `tokenizer/` vs `model/`). A plain JSON file next to
`model.bin`:

```json
{
  "char_to_id": {"\n": 0, " ": 1, "!": 2, ...},
  "id_to_char": ["\n", " ", "!", ...]
}
```

## Versioning

Bump `format_version` on any breaking layout change. The loader should reject files
whose `magic` doesn't match and whose `format_version` it doesn't recognize, rather
than guessing.
