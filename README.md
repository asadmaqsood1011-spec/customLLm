# GPT From Scratch

A decoder-only language model built from first principles in PyTorch and taken
through the **entire modern LLM lifecycle** — tokenizer, pretraining,
supervised fine-tuning, and preference alignment (DPO) — plus a live demo. No
`transformers`, no pretrained weights. PyTorch supplies tensors, autograd and
CUDA; the architecture and every training/alignment algorithm are written by
hand so the math is visible.

A **12M-parameter** model trained on a single RTX 4070 in minutes, that
**beats GPT-2 (124M) by 14% on in-domain bits-per-byte** and demonstrably
improves at instruction-following through SFT and DPO.

→ **Full results and method: [REPORT.md](REPORT.md)**

## Pipeline

```
 BPE tokenizer ─▶ Pretrain (TinyStories) ─▶ SFT (instruction tune) ─▶ DPO (align) ─▶ Demo
   bpe.py            train.py                   sft.py                  dpo.py        app.py
```

## What's implemented from scratch

- **Byte-level BPE tokenizer** (`bpe.py`) — merge-based, regex pre-split, word
  de-dup for speed, special-token support for the chat format.
- **Modern transformer** (`model.py`) — **RoPE** rotary positions, **RMSNorm**,
  **SwiGLU** FFN, **grouped-query attention (GQA)**, **KV-cache** decoding,
  weight tying. Attention/RoPE/norm math written out explicitly.
- **Pretraining** (`train.py`) — AdamW, warmup + cosine LR, grad clip, bf16
  autocast, memmap loading, best-checkpoint saving.
- **Supervised fine-tuning** (`sft.py`) — chat template with **prompt loss
  masking** (train on responses only).
- **DPO** (`dpo.py`) — Direct Preference Optimization from the loss equation,
  with a frozen reference model.
- **Eval + benchmark** (`eval.py`, `benchmark.py`) — instruction-following
  metrics across stages, perplexity, throughput/MFU, and a tokenizer-fair
  bits-per-byte comparison vs pretrained GPT-2.
- **Live demo** (`app.py`) — Gradio chat UI; type a few words, the model writes
  a story.

## Headline results (RTX 4070, 12M params)

| metric | result |
|---|---|
| val perplexity (TinyStories) | 7.84 |
| bits-per-byte vs **GPT-2 124M** | **0.737 vs 0.857 — ours wins by 14%** |
| instruction word hit-rate (base → SFT → DPO) | 0.13 → 0.24 → **0.33** |
| DPO preference accuracy | 0.62 → **1.00** |
| training throughput / MFU | 152k tok/s / 21% |

## Quickstart

```bash
pip install -r requirements.txt

python data.py            # download TinyStories, train BPE, tokenize
python train.py           # pretrain         -> out/ckpt.pt
python sft.py             # instruction tune -> out/sft.pt
python prepare_dpo.py     # build preference pairs
python dpo.py             # DPO align        -> out/dpo.pt

python eval.py            # base vs SFT vs DPO metrics
python benchmark.py --gpt2
python app.py             # live demo at http://127.0.0.1:7860
```

Generate from the CLI:

```bash
python generate.py --prompt "Once upon a time" --max_new_tokens 200
```

## Deploy the demo (Hugging Face Spaces)

1. Create a Space → SDK **Gradio**.
2. Push this repo plus `out/dpo.pt` and `data/tokenizer.json` (and `app.py`,
   `model.py`, `bpe.py`, `chat.py`, `infer.py`).
3. The Space runs `app.py` automatically. CPU tier is fine (12M model).

## Default config

| layers | heads | kv-heads | embed | context | vocab | precision |
|--------|-------|----------|-------|---------|-------|-----------|
| 6      | 6     | 6        | 384   | 256     | 4099  | bf16      |

Tune via flags, e.g. `python train.py --n_layer 8 --n_embd 512`.

## Why this project

To understand exactly how modern LLMs work — not by calling an API, but by
implementing the whole stack: how text becomes tokens, how RoPE/attention mix
information, how a base model is pretrained, then aligned to follow
instructions via SFT and DPO, and how to measure each step honestly. See
[REPORT.md](REPORT.md) for the full write-up and limitations.
