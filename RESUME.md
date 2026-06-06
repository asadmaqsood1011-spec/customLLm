# Resume — project entry

Copy/adapt as needed. Pick the tier that fits your space.

---

## One-liner (skills/header)

**HalluGuard — LLM hallucination detector & GPT built from scratch** · PyTorch,
from-scratch transformer (RoPE/RMSNorm/SwiGLU/GQA), pretraining, SFT, DPO,
classification. [github.com/asadmaqsood1011-spec/customLLm](https://github.com/asadmaqsood1011-spec/customLLm)

---

## Project block (recommended)

**GPT & Hallucination Detector — from scratch in PyTorch** · *personal project*

- Built a decoder-only LLM end-to-end with **no ML frameworks beyond tensors/
  autograd** — byte-level BPE tokenizer, a modern transformer (RoPE, RMSNorm,
  SwiGLU, grouped-query attention, KV-cache), pretraining, supervised
  fine-tuning, and **DPO** preference alignment.
- Shipped **HalluGuard**, a 12M-param detector that flags when an LLM answer is
  **unsupported by its source** — **93.5% accuracy / 0.97 AUROC** on
  length-controlled HaluEval, running locally at **~2 ms/check ($0, fully
  private)** vs ~0.5–1.5 s and per-call cost for a GPT-4 guardrail.
- **Found and corrected a benchmark artifact**: a claim-length heuristic alone
  scored 0.98 AUROC on HaluEval; built length-controlled splits that collapse the
  shortcut, proving the model learned genuine faithfulness signal (not the
  artifact).
- Pretrained model reaches **0.74 bits-per-byte, beating GPT-2 (124M) by 14%**
  on in-domain text; trains in ~5 min on a single RTX 4070 (bf16), ~21% MFU with
  hand-written attention.

---

## Tightest 2-bullet version

- Built a GPT-style LLM **from scratch** in PyTorch (BPE, RoPE/RMSNorm/SwiGLU/GQA
  transformer, pretraining, SFT, DPO) — pretrained model beats GPT-2 124M by 14%
  bits-per-byte in-domain.
- Turned it into **HalluGuard**, a local hallucination detector (**93.5% acc /
  0.97 AUROC**, ~2 ms/check, $0) and **identified + corrected a length artifact**
  in the HaluEval benchmark.

---

## Talking points (for interviews)

- Why bits-per-byte, not perplexity, to compare across tokenizers.
- The DPO loss and why a frozen reference model is needed.
- Faithfulness vs factuality — why only the former is feasible at 12M params.
- How you caught the length artifact and why controlling for it matters.
- Honest limit: in-distribution supervised result; cross-domain generalization
  and in-domain pretraining are the next steps.
