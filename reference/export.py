"""
Export a trained TinyTransformer checkpoint to the custom `model.bin` format defined
in docs/model_format.md. This is the ONLY place that format's byte layout is written
from the Python side; the eventual C++ loader (runtime/model/model_loader) must match
it exactly. Keep this file and docs/model_format.md in sync.

Usage:
    python export.py --ckpt ../models/tiny_transformer.pt --out ../models/model.bin
"""
import argparse
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from model import TinyTransformerConfig

HERE = Path(__file__).parent

MAGIC = 0x544C4D52  # "TLMR" little-endian
FORMAT_VERSION = 1
ARCH_TINY_TRANSFORMER_DECODER = 0
DTYPE_FP32 = 0

HEADER_SIZE = 64
ENTRY_NAME_SIZE = 64
ENTRY_SIZE = 64 + 4 + 4 + 16 + 8 + 8  # name + dtype + ndim + shape[4] + offset + nbytes = 104


def linear_weight_transposed(state_dict, key: str) -> np.ndarray:
    """PyTorch nn.Linear.weight is [out, in]; model_format.md stores [in, out] so the
    runtime can compute Y = X @ W + b without transposing at load time."""
    w = state_dict[key].detach().cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(w.T)


def collect_tensors(state_dict, cfg: TinyTransformerConfig) -> List[Tuple[str, np.ndarray]]:
    tensors: List[Tuple[str, np.ndarray]] = []

    tensors.append(("tok_embedding.weight",
                     state_dict["tok_embedding.weight"].detach().cpu().numpy().astype(np.float32)))
    tensors.append(("pos_embedding.weight",
                     state_dict["pos_embedding.weight"].detach().cpu().numpy().astype(np.float32)))

    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        tensors.append((f"layers.{i}.ln1.weight", state_dict[p + "ln1.weight"].numpy().astype(np.float32)))
        tensors.append((f"layers.{i}.ln1.bias", state_dict[p + "ln1.bias"].numpy().astype(np.float32)))

        tensors.append((f"layers.{i}.attn.wq.weight", linear_weight_transposed(state_dict, p + "attn.wq.weight")))
        tensors.append((f"layers.{i}.attn.wq.bias", state_dict[p + "attn.wq.bias"].numpy().astype(np.float32)))
        tensors.append((f"layers.{i}.attn.wk.weight", linear_weight_transposed(state_dict, p + "attn.wk.weight")))
        tensors.append((f"layers.{i}.attn.wk.bias", state_dict[p + "attn.wk.bias"].numpy().astype(np.float32)))
        tensors.append((f"layers.{i}.attn.wv.weight", linear_weight_transposed(state_dict, p + "attn.wv.weight")))
        tensors.append((f"layers.{i}.attn.wv.bias", state_dict[p + "attn.wv.bias"].numpy().astype(np.float32)))
        tensors.append((f"layers.{i}.attn.wo.weight", linear_weight_transposed(state_dict, p + "attn.wo.weight")))
        tensors.append((f"layers.{i}.attn.wo.bias", state_dict[p + "attn.wo.bias"].numpy().astype(np.float32)))

        tensors.append((f"layers.{i}.ln2.weight", state_dict[p + "ln2.weight"].numpy().astype(np.float32)))
        tensors.append((f"layers.{i}.ln2.bias", state_dict[p + "ln2.bias"].numpy().astype(np.float32)))

        tensors.append((f"layers.{i}.mlp.fc1.weight", linear_weight_transposed(state_dict, p + "mlp.fc1.weight")))
        tensors.append((f"layers.{i}.mlp.fc1.bias", state_dict[p + "mlp.fc1.bias"].numpy().astype(np.float32)))
        tensors.append((f"layers.{i}.mlp.fc2.weight", linear_weight_transposed(state_dict, p + "mlp.fc2.weight")))
        tensors.append((f"layers.{i}.mlp.fc2.bias", state_dict[p + "mlp.fc2.bias"].numpy().astype(np.float32)))

    tensors.append(("ln_f.weight", state_dict["ln_f.weight"].numpy().astype(np.float32)))
    tensors.append(("ln_f.bias", state_dict["ln_f.bias"].numpy().astype(np.float32)))
    tensors.append(("lm_head.weight", linear_weight_transposed(state_dict, "lm_head.weight")))
    tensors.append(("lm_head.bias", state_dict["lm_head.bias"].numpy().astype(np.float32)))

    return tensors


def write_model_bin(out_path: Path, cfg: TinyTransformerConfig, tensors: List[Tuple[str, np.ndarray]]) -> None:
    tensor_count = len(tensors)
    tensor_table_offset = HEADER_SIZE
    data_section_offset = tensor_table_offset + tensor_count * ENTRY_SIZE

    # compute per-tensor offsets (relative to data_section_offset) and validate shapes
    entries = []
    data_blobs = []
    running_offset = 0
    for name, arr in tensors:
        if name.encode("utf-8").__len__() >= ENTRY_NAME_SIZE:
            raise ValueError(f"tensor name too long for format: {name!r}")
        if arr.ndim > 4:
            raise ValueError(f"tensor {name!r} has ndim={arr.ndim} > 4")
        arr = np.ascontiguousarray(arr.astype(np.float32))
        nbytes = arr.nbytes
        shape = list(arr.shape) + [0] * (4 - arr.ndim)
        entries.append((name, arr.ndim, shape, running_offset, nbytes))
        data_blobs.append(arr)
        running_offset += nbytes

    with open(out_path, "wb") as f:
        # header
        f.write(struct.pack(
            "<IIIIIIIIIIII16s",
            MAGIC,
            FORMAT_VERSION,
            ARCH_TINY_TRANSFORMER_DECODER,
            cfg.vocab_size,
            cfg.d_model,
            cfg.n_layers,
            cfg.n_heads,
            cfg.d_ff,
            cfg.context_length,
            tensor_count,
            tensor_table_offset,
            data_section_offset,
            b"\x00" * 16,
        ))

        # tensor table
        for name, ndim, shape, offset, nbytes in entries:
            name_bytes = name.encode("utf-8").ljust(ENTRY_NAME_SIZE, b"\x00")
            f.write(struct.pack(
                f"<{ENTRY_NAME_SIZE}sIIIIIIQQ",  # name, dtype, ndim, shape[4], offset, nbytes
                name_bytes, DTYPE_FP32, ndim,
                shape[0], shape[1], shape[2], shape[3],
                offset, nbytes,
            ))

        # data section
        for arr in data_blobs:
            f.write(arr.tobytes(order="C"))

    total_params = sum(arr.size for _, arr in tensors)
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {tensor_count} tensors, "
          f"{total_params:,} params)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(HERE / ".." / "models" / "tiny_transformer.pt"))
    parser.add_argument("--out", default=str(HERE / ".." / "models" / "model.bin"))
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Run `python train.py` first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TinyTransformerConfig(**ckpt["config"])
    state_dict = ckpt["model_state_dict"]

    tensors = collect_tensors(state_dict, cfg)
    write_model_bin(Path(args.out), cfg, tensors)


if __name__ == "__main__":
    main()
