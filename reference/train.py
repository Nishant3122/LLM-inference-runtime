"""
Train the Stage-1 tiny char-level Transformer on the synthetic corpus.

This is intentionally a short, CPU-friendly training run (see docs/architecture.md:
training a good language model is explicitly a non-goal of this project — we only
need a model whose forward pass is non-trivial enough to be a meaningful correctness
target for the C++/CUDA runtime).

Usage:
    python data/make_corpus.py     # once, to generate data/corpus.txt
    python train.py
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TinyTransformer, TinyTransformerConfig
from tokenizer import CharTokenizer

HERE = Path(__file__).parent


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(HERE / "data" / "corpus.txt"))
    parser.add_argument("--out-dir", default=str(HERE / ".." / "models"))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run `python data/make_corpus.py` first."
        )
    text = corpus_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.block_size,
    )
    model = TinyTransformer(cfg).to(args.device)
    print(f"Model: {cfg}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Device: {args.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "vocab.json"))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                  f"| {elapsed:.1f}s")

    ckpt_path = out_dir / "tiny_transformer.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved vocab to {out_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()
