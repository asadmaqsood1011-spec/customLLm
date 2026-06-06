"""Evaluate the faithfulness classifier honestly.

Reports our model's accuracy / precision / recall / F1 / AUROC on the held-out
test set, plus **naive baselines** (lexical overlap, claim length) to check
whether the benchmark can be gamed by a trivial cue — and per-check latency for
the cost story.

    python eval_cls.py
"""

import argparse
import math
import os
import re
import time

import numpy as np
import torch

import chat
import hallu_data as hd
from bpe import BPETokenizer
from classifier import GPTClassifier
from model import GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def prf(pred, labels):
    pred, labels = np.asarray(pred), np.asarray(labels)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (pred == labels).mean()
    return acc, prec, rec, f1


def overlap_score(source, claim):
    """1 - lexical overlap: high => claim words missing from source (cue for hallucination)."""
    s = set(re.findall(r"[a-z]+", source.lower()))
    c = re.findall(r"[a-z]+", claim.lower())
    if not c:
        return 0.0
    return 1.0 - sum(w in s for w in c) / len(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(OUT_DIR, "halluguard.pt"))
    p.add_argument("--tag", default="", help="split prefix, e.g. 'ctrl_'")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else \
        torch.autocast(device_type="cpu", enabled=False)

    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    pad_id = tok.special_tokens[chat.EOT]
    ck = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ck["config"])
    bs = ck["block_size"]
    model = GPTClassifier(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    test = hd.load_split(args.tag + "test")
    labels = [r["label"] for r in test]

    # --- our model ---
    probs, t0 = [], time.time()
    with torch.no_grad():
        for s in range(0, len(test), 64):
            X, L, _ = hd.make_batch(tok, test[s:s + 64], bs, pad_id, device)
            with ctx:
                logit, _ = model(X, L)
            probs.extend(torch.softmax(logit.float(), -1)[:, 1].tolist())
    dt = time.time() - t0
    preds = [int(p > 0.5) for p in probs]
    acc, prec, rec, f1 = prf(preds, labels)

    print(f"test examples: {len(test):,}\n")
    print("== HalluGuard (12M, from scratch) ==")
    print(f"  accuracy {acc:.4f} | precision {prec:.4f} | recall {rec:.4f} | F1 {f1:.4f}")
    print(f"  AUROC    {auroc(probs, labels):.4f}")
    print(f"  latency  {dt/len(test)*1000:.2f} ms/check ({len(test)/dt:,.0f} checks/sec, {device})\n")

    # --- naive baselines (artifact check) ---
    ov = [overlap_score(r["source"], r["claim"]) for r in test]
    ln = [len(r["claim"]) for r in test]
    print("== Naive baselines (can the benchmark be gamed?) ==")
    print(f"  lexical-overlap cue  AUROC {auroc(ov, labels):.4f}")
    print(f"  claim-length cue     AUROC {auroc(ln, labels):.4f}")
    print("  (AUROC near 0.5 = no trivial cue; high = benchmark has a shortcut)\n")

    # --- cost framing vs GPT-4 ---
    ms = dt / len(test) * 1000
    print("== Cost vs GPT-4 as a per-response guardrail ==")
    print(f"  ours : ~{ms:.1f} ms/check, $0 (local), private")
    print(f"  GPT-4: ~500-1500 ms/check, ~$0.005-0.02/check (API), data leaves device")
    print("  reference: zero-shot ChatGPT on HaluEval-QA ~65% acc (Li et al. 2023)")


if __name__ == "__main__":
    main()
