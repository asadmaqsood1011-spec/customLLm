"""Train the GPT. Reads data/train.bin & val.bin, writes checkpoints to out/.

    python train.py                 # defaults, CUDA + bf16 if available
    python train.py --max_iters 1000 --batch_size 32
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def get_batch(split, block_size, batch_size, device):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    # memmap so we don't load the whole corpus into RAM each call
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_at(it, args):
    """Linear warmup then cosine decay to a 10% floor."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    if it > args.max_iters:
        return args.lr * 0.1
    ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.lr * 0.1 + coeff * (args.lr - args.lr * 0.1)


@torch.no_grad()
def estimate_loss(model, args, device, ctx):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="torch.compile (faster, slow first step)")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" \
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device={device} dtype={dtype}")

    # vocab_size from tokenizer
    from bpe import BPETokenizer
    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"model params: {model.num_params()/1e6:.2f}M")
    if args.compile:
        model = torch.compile(model)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = lr_at(it, args)
        for g in optim.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, args, device, ctx)
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {lr:.2e} | {dt:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                     "config": cfg.__dict__, "iter": it, "val_loss": best_val},
                    os.path.join(OUT_DIR, "ckpt.pt"),
                )

        if it == args.max_iters:
            break

        x, y = get_batch("train", args.block_size, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

    print(f"done. best val loss {best_val:.4f}. checkpoint in {OUT_DIR}/ckpt.pt")


if __name__ == "__main__":
    main()
