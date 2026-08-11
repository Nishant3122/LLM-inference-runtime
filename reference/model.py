"""
Stage-1 reference model: a tiny, char-level, decoder-only Transformer.

This is the ground truth for G1 (correctness) in docs/architecture.md: every later
C++/CUDA runtime path (CPU naive -> CPU optimized -> CUDA naive -> CUDA optimized ->
quantized -> adaptive) is validated against this model's forward pass, not against
"what a Transformer generally does". Keep this file boring and explicit — clever
implementations here would make it a worse reference.

Architecture (pre-norm, matches docs/architecture.md §5 and the original spec's
block diagram):

    x = tok_embedding[ids] + pos_embedding[positions]
    for each layer:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    x = LayerNorm(x)
    logits = LMHead(x)

Attention is computed explicitly (Q = XWq, K = XWk, V = XWv, softmax(QK^T/sqrt(d))V)
rather than via a fused kernel, per the original spec's instruction to implement the
naive form first and only optimize after correctness is established (§11). That
instruction is really aimed at the C++/CUDA ports, but keeping the PyTorch reference
just as explicit makes it easy to hand-verify against runtime.py output tensor-by-tensor.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyTransformerConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Explicit multi-head causal self-attention: Q=XWq, K=XWk, V=XWv,
    softmax(QK^T / sqrt(head_dim)) V, then output projection."""

    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # causal mask, precomputed up to context_length
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim

        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)  # [B, H, T, T]
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v  # [B, H, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.wo(out))


class MLP(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] int64 token ids -> logits: [B, T, vocab_size]."""
        B, T = ids.shape
        assert T <= self.cfg.context_length, (
            f"sequence length {T} exceeds context_length {self.cfg.context_length}"
        )
        positions = torch.arange(T, device=ids.device).unsqueeze(0)  # [1, T]
        x = self.tok_embedding(ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
