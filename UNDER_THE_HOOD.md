# Under the Hood — how the whole system works

A plain-language walkthrough of every layer, from raw bytes to a hallucination
detector. Everything here is built from scratch in PyTorch (tensors + autograd +
CUDA only); no `transformers`, no pretrained weights, no LangChain.

```
text ─▶ BPE tokenizer ─▶ Transformer (pretrain) ─▶ SFT ─▶ DPO ─▶ + classifier head ─▶ HalluGuard
        bpe.py            model.py / train.py      sft.py  dpo.py   classifier.py        (the product)
```

---

## 1. Tokenizer — turning text into integers (`bpe.py`)

Computers see numbers, not text. Byte-level **BPE** (Byte Pair Encoding):

1. Start from raw UTF-8 **bytes** → 256 base tokens. Any string is encodable.
2. Repeatedly find the most frequent adjacent pair of tokens and merge it into a
   new token. Do this 3,840 times → vocab of ~4,096.
3. Common chunks (` the`, ` soul`, `ing`) become single tokens; rare text falls
   back to bytes.

We pre-split text into words with a regex and count *unique* words, so training
is fast. Result: **~4× compression** (4 characters per token on average).
Special tokens (`<|endoftext|>`, `<|user|>`, `<|assistant|>`) get reserved ids
above the BPE range.

## 2. The model — a modern transformer (`model.py`)

A **decoder-only transformer**: it reads a sequence of tokens and predicts the
next one. Built with current (Llama-style) components, not 2019 GPT-2 ones:

- **Token embedding** — each token id → a 384-dim vector.
- **RoPE (rotary positions)** — instead of a learned position table, query/key
  vectors are *rotated* by an angle proportional to their position. This encodes
  *relative* distance directly in attention and extrapolates to longer contexts.
- **Causal self-attention** — every token computes Query/Key/Value vectors;
  attention score = `softmax(Q·Kᵀ / √d)`, masked so a token can only see tokens
  *before* it. This is how information moves across the sequence. **GQA**
  (grouped-query attention) lets several query heads share one key/value head to
  shrink the cache.
- **RMSNorm** — normalizes vectors by their root-mean-square (cheaper, stabler
  than LayerNorm), applied before each sub-layer.
- **SwiGLU feed-forward** — a gated MLP: `down(silu(gate(x)) · up(x))`. Stronger
  than a plain GELU MLP at the same parameter count.
- **Residual connections** — each block adds its output back to its input, so
  gradients flow and layers refine rather than replace.
- **Weight tying** — input embedding and output projection share weights.
- **KV cache** — during generation, past Key/Value vectors are cached so each new
  token costs O(1) instead of reprocessing the whole sequence.

Stacking 6 of these blocks (6 heads, 384-dim) = **12.2M parameters**.

## 3. Pretraining — learning language (`train.py`, `data.py`)

The model is shown billions of token windows from **TinyStories** (simple
synthetic children's stories, chosen because small models can actually learn
coherent text from them) and trained to predict the next token. Loss =
**cross-entropy** between predicted and actual next token.

Mechanics: AdamW optimizer, learning-rate **warmup then cosine decay**, gradient
clipping, **bf16** mixed precision, data streamed via memmap. After ~5k steps
(~5 min on an RTX 4070): validation loss **2.06**, perplexity **7.84**,
**0.737 bits-per-byte — 14% better than GPT-2 (124M) on in-domain text.** It now
writes grammatical, on-topic short stories.

## 4. Supervised fine-tuning (SFT) — learning to follow instructions (`sft.py`)

A pretrained model continues text; it doesn't *answer*. SFT shows it
instruction→response examples formatted as a chat turn:

```
<|user|> Write a short story using these words: dog, ball, happy. <|assistant|> <story> <|endoftext|>
```

Key trick: **loss is masked on the prompt** (labels = −1 there) — the model is
trained *only* on the response tokens, so it learns to respond, not to echo the
instruction. Effect: instruction word-use jumps from 0.13 → 0.24.

## 5. DPO — aligning to preferences (`dpo.py`)

Even after SFT, some responses are better than others. **Direct Preference
Optimization** nudges the model toward preferred answers without a reward model
or reinforcement learning. For each `(prompt, chosen, rejected)` triple:

```
loss = −log σ( β · [ (logπ_chosen − logπ_rejected) − (logπ_ref_chosen − logπ_ref_rejected) ] )
```

`π` is our model; `π_ref` is a **frozen copy** of the SFT model that anchors us
so we don't drift. In words: *raise the probability of the chosen answer relative
to the rejected one, measured against the frozen reference.* Preference pairs
were built by sampling the SFT model twice and ranking with a transparent reward
(word-use + anti-repetition + length). Result: preference accuracy 0.62 → **1.00**,
and instruction word-use rises again to **0.33** while diversity recovers.

## 6. HalluGuard — the product (`classifier.py`, `train_cls.py`)

The real problem: deployed LLMs **make things up**. The most useful, checkable
version is **faithfulness** — did an answer stay true to its source? (Checking
truth *about the world* needs knowledge a 12M model doesn't have; checking
*against a given source* does not — so that's what we build.)

**How it works:**
1. Reuse the transformer **backbone**, but replace the next-token head with a
   small **classification head** on the last token's hidden state.
2. Feed it `<|user|> {source} <|assistant|> {claim}` → 2 logits:
   **supported** vs **hallucinated**.
3. Fine-tune on **HaluEval** (knowledge + correct answer = supported; knowledge +
   fabricated answer = hallucinated).

**The honest part — finding and fixing a benchmark artifact:**
A first model scored 98%. Suspicious. A naive **claim-length** heuristic alone
scored **0.976 AUROC** — HaluEval's fake answers are simply *longer*. So the high
score was mostly a shortcut, not understanding. We built **length-controlled**
splits (matched length distributions per class), collapsing the length cue. On
that fair data the model still reaches **93.5% accuracy / 0.97 AUROC** — real
signal. (An ablation showed TinyStories pretraining gave no significant lift here
— domain mismatch; the task is learnable at this scale regardless.)

**Why a tiny model is the right tool:** this runs as a guardrail on *every* LLM
response, so it must be cheap. HalluGuard: **~2 ms/check, $0, fully local
(private)**. A GPT-4 guardrail: ~0.5–1.5 s and a per-call fee, with data leaving
the device.

## TL;DR

Bytes → BPE tokens → a from-scratch modern transformer → pretrained on
TinyStories → instruction-tuned (SFT) → preference-aligned (DPO) → re-headed into
**HalluGuard**, a local, ~2 ms, $0 faithfulness detector that flags when an AI
answer isn't supported by its source — with the benchmark artifact found and
controlled for. Every layer hand-built.
