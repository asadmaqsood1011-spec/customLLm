# GPT From Scratch → HalluGuard

A decoder-only LLM built from first principles in PyTorch and taken through the
**entire modern lifecycle** — tokenizer, pretraining, SFT, DPO — then re-headed
into **HalluGuard**, a hallucination detector that solves a real problem:
**flagging when an AI answer isn't supported by its source.** No `transformers`,
no pretrained weights — every layer hand-built.

**HalluGuard** runs as a guardrail on *every* LLM response, so it's tiny, local
and free where a GPT-4 check is slow and paid:

| | result |
|---|---|
| accuracy / AUROC (length-controlled HaluEval) | **93.5% / 0.97** |
| latency / cost / privacy | **~2 ms/check · $0 · fully local** |
| vs GPT-4 guardrail | ~0.5–1.5 s, per-call fee, data leaves device |

It also **catches a benchmark artifact**: a claim-length heuristic alone scores
0.98 AUROC on raw HaluEval, so the project builds length-controlled splits that
collapse the shortcut and prove the model learned real signal.

The underlying 12M model also **beats GPT-2 (124M) by 14% on in-domain
bits-per-byte**, trained in ~5 min on one RTX 4070.

→ **Full method and results: [REPORT.md](REPORT.md)**

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

Train **HalluGuard** (the hallucination detector):

```bash
python prepare_halueval.py   # download HaluEval, build labeled (source, claim) pairs
python make_controlled.py    # length-controlled splits (remove the artifact)
python train_cls.py --tag ctrl_ --out out/halluguard_ctrl.pt   # fine-tune detector
python eval_cls.py  --tag ctrl_ --ckpt out/halluguard_ctrl.pt  # accuracy/F1/AUROC + baselines
```

Audit the benchmarks for shortcuts, then run HalluGuard as a guardrail:

```bash
python prepare_benchmarks.py   # download 3 source-grounded HaluEval subsets
python shortcuts.py            # AUROC of model-free cues per benchmark
python make_controlled.py      # length-matched splits that kill the cue
python audit_report.py         # write FINDINGS.md from the measured numbers

python guard.py                # one-shot demo of the check(source, claim) API
python guard_app.py            # Gradio demo: paste a source + an answer
```

A faithfulness benchmark should test whether a claim is supported by its source.
`shortcuts.py` shows all three HaluEval subsets leak the label through claim
length (a model that just counts characters scores up to 0.98 AUROC on QA).
`make_controlled.py` matches the per-class length distribution so that cue dies,
and the detector is re-evaluated on the honest splits. Full write-up in
[FINDINGS.md](FINDINGS.md).

Generate from the CLI:

```bash
python generate.py --prompt "Once upon a time" --max_new_tokens 200
```

## Tests

A small pytest suite covers the parts most likely to break silently:

- `tests/test_bpe.py` checks encode/decode roundtrips (ASCII, unicode and emoji),
  special-token splicing, and save/load.
- `tests/test_model.py` checks output shapes, causal masking, grouped-query
  attention, and that cached decoding matches a full forward pass token for token.

```bash
pip install pytest
pytest -q
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
