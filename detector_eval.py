"""Evaluate a trained HalluGuard checkpoint on a controlled test split.

Factored out of eval_cls.py so the audit report can fold the detector's numbers
into the same document as the shortcut baselines, always recomputed from the
checkpoint rather than copied by hand.
"""

import os

import torch

import chat
import hallu_data as hd
from bpe import BPETokenizer
from classifier import GPTClassifier
from eval_cls import auroc, prf
from model import GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def evaluate(split, ckpt, device=None):
    """Run a checkpoint over hallu_<split>_test; return acc/precision/recall/F1/AUROC."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    pad_id = tok.special_tokens[chat.EOT]
    ck = torch.load(ckpt, map_location=device)
    cfg = GPTConfig(**ck["config"])
    bs = ck["block_size"]
    model = GPTClassifier(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = hd.load_split(f"{split}_test")
    labels = [r["label"] for r in rows]
    probs = []
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            X, L, _ = hd.make_batch(tok, rows[s:s + 64], bs, pad_id, device)
            logit, _ = model(X, L)
            probs.extend(torch.softmax(logit.float(), -1)[:, 1].tolist())
    preds = [int(p > 0.5) for p in probs]
    acc, prec, rec, f1 = prf(preds, labels)
    return {"n": len(rows), "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auroc": auroc(probs, labels)}
