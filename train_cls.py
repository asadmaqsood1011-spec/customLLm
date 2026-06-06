"""Fine-tune the from-scratch transformer into a faithfulness classifier.

Loads the pretrained LM backbone (out/ckpt.pt), attaches a classification head,
and trains on HaluEval to detect whether a claim is faithful to its source.

    python train_cls.py                      # -> out/halluguard.pt
    python train_cls.py --scratch            # ablation: no pretrained backbone
"""

import argparse
import os
import random
import time

import torch

import chat
import hallu_data as hd
from bpe import BPETokenizer
from classifier import GPTClassifier
from model import GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


@torch.no_grad()
def accuracy(model, tok, rows, block_size, pad_id, device, ctx, bs=64):
    model.eval()
    correct = 0
    for s in range(0, len(rows), bs):
        X, L, Y = hd.make_batch(tok, rows[s:s + bs], block_size, pad_id, device)
        with ctx:
            logits, _ = model(X, L)
        correct += (logits.argmax(-1) == Y).sum().item()
    model.train()
    return correct / len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--scratch", action="store_true", help="skip pretrained backbone (ablation)")
    p.add_argument("--tag", default="", help="split prefix, e.g. 'ctrl_' for length-controlled")
    p.add_argument("--out", default=os.path.join(OUT_DIR, "halluguard.pt"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else \
        torch.autocast(device_type="cpu", enabled=False)

    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    pad_id = tok.special_tokens[chat.EOT]

    ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    cfg.block_size = args.block_size          # longer context for sources
    cfg.dropout = 0.1

    model = GPTClassifier(cfg).to(device)
    if args.scratch:
        print("ABLATION: training classifier from random init (no pretraining)")
    else:
        miss, unexp = model.load_backbone(ckpt["model"])
        print(f"loaded pretrained backbone (skipped {len(unexp)} LM-head tensors)")

    train, val = hd.load_split(args.tag + "train"), hd.load_split(args.tag + "val")
    print(f"train {len(train):,} | val {len(val):,} | block_size {args.block_size}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    best, t0 = 0.0, time.time()
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        for s in range(0, len(train) - args.batch_size, args.batch_size):
            X, L, Y = hd.make_batch(tok, train[s:s + args.batch_size], args.block_size, pad_id, device)
            with ctx:
                _, loss = model(X, L, Y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        acc = accuracy(model, tok, val, args.block_size, pad_id, device, ctx)
        print(f"epoch {epoch} | val acc {acc:.4f} | {time.time()-t0:.0f}s")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                        "block_size": args.block_size, "val_acc": best}, args.out)
    print(f"done. best val acc {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
