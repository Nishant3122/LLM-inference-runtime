"""
Dump "golden" reference outputs from the trained PyTorch model: fixed prompts ->
exact logits, saved as .npy under tests/golden/. This is the artifact G1 (correctness,
docs/architecture.md §4) is checked against once the C++ runtime exists — Phase 1's
first real milestone is `cpp_logits ≈ golden_logits` within a small numerical tolerance
(see docs/architecture.md Engineering Principle 1: correct before optimized).

Also dumps intermediate activations (after embedding, after each block, after final
norm) for a single short prompt, so a failing comparison in Phase 1 can be bisected to
the specific op (embedding vs. attention vs. MLP vs. norm) instead of only comparing
final logits.

Usage:
    python dump_golden.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from model import TinyTransformer, TinyTransformerConfig
from tokenizer import CharTokenizer

HERE = Path(__file__).parent
MODELS_DIR = HERE / ".." / "models"
GOLDEN_DIR = HERE / ".." / "tests" / "golden"

# Fixed, short, deterministic prompts. Kept short so activation dumps stay small and
# so a human can trace through the runtime's intermediate tensors by hand if needed.
GOLDEN_PROMPTS = [
    "The cat",
    "A small bird finds",
]


def main():
    ckpt_path = MODELS_DIR / "tiny_transformer.pt"
    vocab_path = MODELS_DIR / "vocab.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Run `python train.py` first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TinyTransformerConfig(**ckpt["config"])
    model = TinyTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.from_json(str(vocab_path))
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"config": cfg.__dict__, "cases": []}

    with torch.no_grad():
        for case_idx, prompt in enumerate(GOLDEN_PROMPTS):
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

            # re-run forward manually to capture intermediates (mirrors model.forward)
            positions = torch.arange(ids.shape[1]).unsqueeze(0)
            x = model.tok_embedding(ids) + model.pos_embedding(positions)
            activations = {"embedding_output": x.clone()}
            for i, layer in enumerate(model.layers):
                x = layer(x)
                activations[f"block_{i}_output"] = x.clone()
            x = model.ln_f(x)
            activations["final_norm_output"] = x.clone()
            logits = model.lm_head(x)
            activations["logits"] = logits.clone()

            case_name = f"case_{case_idx}"
            case_dir = GOLDEN_DIR / case_name
            case_dir.mkdir(exist_ok=True)
            (case_dir / "input_ids.npy").write_bytes(
                np.asarray(ids[0].tolist(), dtype=np.int32).tobytes()
            )
            for key, tensor in activations.items():
                np.save(case_dir / f"{key}.npy", tensor.squeeze(0).numpy().astype(np.float32))

            next_token_id = int(torch.argmax(logits[0, -1]).item())
            manifest["cases"].append({
                "name": case_name,
                "prompt": prompt,
                "input_ids": ids[0].tolist(),
                "seq_len": ids.shape[1],
                "greedy_next_token_id": next_token_id,
                "greedy_next_token_char": tokenizer.decode([next_token_id]),
            })

    (GOLDEN_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote golden outputs for {len(GOLDEN_PROMPTS)} prompt(s) to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
