# Technical Report — GPT From Scratch

A decoder-only language model implemented from first principles in PyTorch and
taken through the **full modern LLM lifecycle**: tokenizer → pretraining →
supervised fine-tuning → preference alignment → evaluation. No `transformers`,
no pretrained weights. PyTorch provides tensors, autograd and CUDA; the
architecture, training and alignment algorithms are written by hand.

All numbers below were measured on a single **RTX 4070 (12 GB)**, bf16.

---

## 1. Architecture (`model.py`)

Modern (Llama-style) choices rather than 2019 GPT-2 defaults:

| component | choice | why |
|-----------|--------|-----|
| position  | **RoPE** (rotary) | relative positions, no learned table, extrapolates |
| norm      | **RMSNorm** | cheaper than LayerNorm, no mean/centering, stable |
| FFN       | **SwiGLU** | gated activation, stronger than GELU MLP at equal params |
| attention | **GQA** (grouped-query) | fewer KV heads → smaller KV cache |
| decoding  | **KV cache** | O(1) per-token generation instead of O(T) |
| misc      | weight tying, scaled residual init | GPT-2 tricks that still help |

KV-cache correctness was unit-tested against a full forward pass (max logit
diff `3.6e-7`).

Default model: 6 layers, 6 heads, 384-dim, 256 context → **12.2M params**.

## 2. Tokenizer (`bpe.py`)

Byte-level BPE from scratch (regex pre-split + de-duplicated word counts for
speed; special-token support for the chat format). Vocab 4096 + 3 special
tokens. **4.0× compression** on TinyStories.

## 3. Pretraining (`data.py`, `train.py`)

Corpus: **TinyStories** (Eldan & Li, 2023) — synthetic simple-English stories
explicitly designed so small models can produce coherent text. 60 MB / 15.6M
tokens. AdamW, warmup + cosine LR, grad clip, bf16 autocast.

| | value |
|---|---|
| train / val loss @ 5k iters | 1.92 / **2.06** (tracks closely — no overfit) |
| val perplexity | **7.84** |
| bits-per-byte | **0.737** |
| training throughput | 152k tok/s |
| MFU | 21% (hand-written attention, no flash-attn) |

Sample (`temperature 0.7`):

> *Once upon a time, there was a little girl named Lily. She loved to play
> outside and explore the world around her. One day, she was playing in her
> garden when she noticed a big, scary monster...*

### Fair comparison vs pretrained GPT-2 (124M)

Bits-per-byte on the same held-out text (tokenizer-independent, so comparable
across vocabularies — perplexity is not):

| model | bits-per-byte |
|-------|---------------|
| GPT-2 124M (general) | 0.857 |
| **ours 12M (in-domain)** | **0.737** |

A 12M model trained from scratch **beats GPT-2 by 14%** on in-domain text —
the textbook "small specialized model > large general model on its niche"
result, reproduced honestly.

## 4. Supervised fine-tuning (`sft.py`)

Task: *"Write a short story using these words: …"* — chosen because it's
learnable by a tiny model and **measurable** (does the output contain the
words?). Chat format with `<|user|>` / `<|assistant|>` special tokens.

Key detail: **loss is masked on the prompt** (labels = −1); the model is
trained only on response tokens. This is what teaches it to *respond* rather
than to continue the instruction text.

Effect (same prompt, "dog, ball, happy"):

- **base**: ignores the instruction, continues as generic text — 0/3 words.
- **SFT**: produces a story on topic, uses the words.

## 5. Preference alignment — DPO (`dpo.py`)

DPO (Rafailov et al., 2023) implemented directly from the loss:

```
loss = -log σ( β · [ (logπ_chosen − logπ_rejected) − (logπ_ref_chosen − logπ_ref_rejected) ] )
```

A frozen copy of the SFT model is the reference. Preference pairs (1,200) were
built by sampling two completions from the SFT model and ranking them with a
transparent reward proxy (`word_hit_rate + 0.3·distinct-2 + length bonus`) —
honest stand-in for a learned/human reward model; the DPO math is identical.

Training (β = 0.1):

| step | loss | reward margin | pref. accuracy |
|------|------|---------------|----------------|
| 50   | 0.69 | +0.007 | 0.62 |
| 150  | 0.40 | **+0.73** | **1.00** |

## 6. Evaluation (`eval.py`)

100 held-out prompts (seed unused in training), identical sampling noise per
model:

| stage | word hit-rate | distinct-2 | avg words |
|-------|---------------|------------|-----------|
| base  | 0.132 | 0.881 | 89 |
| SFT   | 0.235 | 0.793 | 118 |
| **DPO** | **0.325** | **0.884** | 122 |

SFT ~doubles instruction-following over base; DPO improves it further **and**
recovers the response diversity SFT had traded away (both were in DPO's
reward). Monotonic improvement on the targeted metrics.

## 7. HalluGuard — faithfulness detector (`classifier.py`, `train_cls.py`)

The model is re-headed into a **classifier** (drop the LM head, attach a 2-class
head on the last token's hidden state) to solve a real problem: **detect when an
LLM answer is unsupported by its source** (faithfulness — feasible at 12M;
world-factuality is not). Trained on **HaluEval-QA** (`knowledge+question` →
`right_answer`=supported / `hallucinated_answer`=hallucinated).

**Benchmark artifact found and corrected.** A first run scored 98% — but a naive
**claim-length** heuristic alone scores **0.976 AUROC**: HaluEval's fake answers
are simply longer. We built **length-controlled** splits (per-class length
distributions matched), which collapses the cue. Honest results on the
controlled test set:

| | accuracy | F1 | AUROC |
|---|---|---|---|
| **HalluGuard (12M)** | **0.935** | **0.935** | **0.973** |
| claim-length cue (was 0.976) | — | — | 0.731 |
| lexical-overlap cue | — | — | 0.871 |

The length shortcut is neutralized yet the model holds 0.97 AUROC → it learned
real signal. (Lexical overlap staying predictive is legitimate — missing source
words genuinely indicate unfaithfulness.)

**Cost story (per-response guardrail):** ~**2 ms/check, $0, fully local** vs a
GPT-4 check at ~0.5–1.5 s + per-call fee + data leaving the device.

**Ablation:** classifier from random init (0.914 val) ≈ from pretrained backbone
(0.909 val) — TinyStories pretraining gave no significant lift here (domain
mismatch with QA text); the task is learnable at this scale either way.
In-domain pretraining is the obvious next improvement.

## 8. Honest limitations

- 12M params on 60 MB is tiny. Absolute instruction-following (0.33 hit-rate)
  is modest — the value here is the **techniques implemented correctly with
  measurable, honestly-reported effects**, not SOTA quality. Every stage moves
  the right metric in the right direction.
- The DPO reward is a heuristic proxy, not a learned reward model (stated above).
- Eval is automatic/metric-based, not human preference.

Scaling any stage (more params, more data, more steps) is a knob, not a
rewrite — the pipeline is the contribution.

## Reproduce

```bash
pip install -r requirements.txt
python data.py            # tokenizer + TinyStories bins
python train.py           # pretrain        -> out/ckpt.pt
python sft.py             # instruction tune -> out/sft.pt
python prepare_dpo.py     # build preference pairs
python dpo.py             # align            -> out/dpo.pt
python eval.py            # base vs SFT vs DPO table
python benchmark.py --gpt2
python app.py             # live demo
```
