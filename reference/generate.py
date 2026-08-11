"""
Sanity-check generation from the trained Stage-1 model: greedy / temperature / top-k,
mirroring the sampling strategies the C++ runtime's `sampling/` module must eventually
replicate (docs/architecture.md §7, spec §21).

Usage:
    python generate.py --prompt "the cat" --max-new-tokens 200 --temperature 0.8 --top-k 20
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TinyTransformer, TinyTransformerConfig
from tokenizer import CharTokenizer

HERE = Path(__file__).parent


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TinyTransformerConfig(**ckpt["config"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str,
             temperature: float = 1.0, top_k: int = 0):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    block_size = model.cfg.context_length

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -block_size:]
        logits = model(ids_cond)[:, -1, :]  # [1, vocab_size], last position only

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(HERE / ".." / "models" / "tiny_transformer.pt"))
    parser.add_argument("--vocab", default=str(HERE / ".." / "models" / "vocab.json"))
    parser.add_argument("--prompt", default="The cat")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"{args.ckpt} not found. Run `python train.py` first.")

    model, cfg = load_model(args.ckpt, args.device)
    tokenizer = CharTokenizer.from_json(args.vocab)

    print(f"Config: {cfg}")
    print(f"Prompt: {args.prompt!r}\n")
    output = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.device,
                       temperature=args.temperature, top_k=args.top_k)
    print(output)


if __name__ == "__main__":
    main()
