"""GPT model, built from scratch in PyTorch — modern (Llama-style) architecture.

Decoder-only transformer with the design choices used by current LLMs rather
than the 2019 GPT-2 ones:

  - **RoPE** rotary position embeddings (no learned positional table)
  - **RMSNorm** instead of LayerNorm
  - **SwiGLU** feed-forward instead of GELU MLP
  - **GQA** grouped-query attention (set n_kv_head < n_head to share KV heads)
  - **KV cache** for fast autoregressive generation

PyTorch supplies tensors, autograd and CUDA; the attention math, RoPE rotation,
norms and caching are written out explicitly.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256       # max context length
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6          # < n_head => grouped-query attention
    n_embd: int = 384
    dropout: float = 0.1
    rope_theta: float = 10000.0
    ffn_hidden: int = 0         # 0 => auto (8/3 * n_embd, rounded to multiple of 64)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # normalize in float32 for stability, then cast back
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.type_as(x)) * self.weight


def build_rope_cache(block_size, head_dim, theta, device=None):
    """Precompute cos/sin tables of shape (block_size, head_dim) for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(block_size).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim) -> broadcast over B, heads
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    """Expand KV heads to match query heads for grouped-query attention."""
    if n_rep == 1:
        return x
    B, kvh, T, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, kvh, n_rep, T, hd)
        .reshape(B, kvh * n_rep, T, hd)
    )


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head

        self.c_q = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, past=None, use_cache=False, start_pos=0):
        B, T, C = x.shape
        hd = self.head_dim

        q = self.c_q(x).view(B, T, self.n_head, hd).transpose(1, 2)      # (B, nh, T, hd)
        k = self.c_k(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)   # (B, nkv, T, hd)
        v = self.c_v(x).view(B, T, self.n_kv_head, hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past is not None:
            past_k, past_v = past
            k = torch.cat([past_k, k], dim=2)   # append along time
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)     # (B, nh, Tq, Tk)
        # causal mask by absolute position: query i (at start_pos+i) sees keys <= its position
        q_pos = (start_pos + torch.arange(T, device=x.device)).unsqueeze(1)   # (Tq,1)
        k_pos = torch.arange(Tk, device=x.device).unsqueeze(0)                # (1,Tk)
        att = att.masked_fill(k_pos > q_pos, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                          # (B, nh, Tq, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * hd)
        return self.resid_dropout(self.c_proj(y)), present


class SwiGLU(nn.Module):
    """Gated feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.ffn_hidden or int(8 / 3 * cfg.n_embd)
        hidden = 64 * ((hidden + 63) // 64)     # round up to multiple of 64
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, past=None, use_cache=False, start_pos=0):
        h, present = self.attn(self.attn_norm(x), cos, sin, past, use_cache, start_pos)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight   # weight tying

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("c_proj.weight", "down.weight")):   # residual projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())   # tied head counted once

    def _forward(self, idx, past_caches=None, start_pos=0, use_cache=False):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, "context overflow"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_caches[i] if past_caches is not None else None
            x, present = block(x, cos, sin, past, use_cache, start_pos)
            if use_cache:
                presents.append(present)
        x = self.norm_f(x)
        return self.head(x), presents

    def forward(self, idx, targets=None):
        logits, _ = self._forward(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Autoregressive sampling using the KV cache (prefill then 1 token/step)."""
        caches, start = None, 0
        cur = idx
        for _ in range(max_new_tokens):
            logits, caches = self._forward(cur, past_caches=caches, start_pos=start, use_cache=True)
            start += cur.size(1)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            cur = next_id
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
