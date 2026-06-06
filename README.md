# GPT From Scratch

A small GPT-style language model implemented from first principles in PyTorch —
including a byte-level BPE tokenizer, the transformer architecture, and the full
training loop. No `transformers`, no pretrained weights. PyTorch supplies only
tensors, autograd, and CUDA; the model is written by hand so every piece of the
math is visible.

Trains on [TinyShakespeare](https://github.com/karpathy/char-rnn) and generates
Shakespeare-like text in a few minutes on a single consumer GPU.

## What's implemented

- **Byte-level BPE tokenizer** (`bpe.py`) — the same merge-based algorithm GPT-2/4
  use, trained from raw UTF-8 bytes. Train / encode / decode / save / load.
- **Decoder-only transformer** (`model.py`) — token + learned positional
  embeddings, multi-head **causal self-attention** with the scaled dot-product
  math written out explicitly, pre-norm residual blocks, GELU MLP, weight tying,
  GPT-2-style scaled initialization.
- **Training loop** (`train.py`) — AdamW, linear warmup + cosine LR decay,
  gradient clipping, mixed-precision (bf16) autocast, `memmap` data loading,
  periodic eval, best-checkpoint saving, optional `torch.compile`.
- **Sampling** (`generate.py`) — autoregressive generation with temperature and
  top-k.

## Architecture

```
tokens ─▶ token emb + positional emb
            │
            ▼
      ┌──────────────┐   ×N blocks
      │  LayerNorm    │
      │  Causal MHSA  │──▶ + residual
      │  LayerNorm    │
      │  MLP (4x,GELU)│──▶ + residual
      └──────────────┘
            │
            ▼
      LayerNorm ─▶ Linear head (tied) ─▶ logits ─▶ softmax
```

## Quickstart

```bash
pip install -r requirements.txt

python data.py        # download + tokenize TinyShakespeare
python train.py       # train (RTX 4070: ~few min for a clean sample)
python generate.py --prompt "ROMEO:" --max_new_tokens 500
```

## Default config

| param        | value |
|--------------|-------|
| layers       | 6     |
| heads        | 6     |
| embedding    | 384   |
| context      | 256   |
| vocab (BPE)  | 4096  |
| precision    | bf16  |

~10M parameters. Tune via flags, e.g. `python train.py --n_layer 8 --n_embd 512`.

## Why this project

Building a transformer end-to-end — tokenizer through training loop — rather than
calling a library, to understand exactly how modern LLMs work: how text becomes
tokens, how attention mixes information across a sequence, and how the model is
optimized.
