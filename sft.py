"""Supervised fine-tuning (instruction tuning) of the pretrained model.

Builds instruction examples from the corpus (see chat.py), formats them as a
chat turn, and trains *only on the response tokens* — the prompt is masked out
of the loss (labels = -1). That masking is the detail that separates real
instruction tuning from "just keep training on concatenated text": we teach the
model to respond, not to generate the instruction.

    python sft.py                 # fine-tune out/ckpt.pt -> out/sft.pt
"""

import argparse
import os
import random
import time

import torch

import chat
from bpe import BPETokenizer
from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def build_examples(tok, n_examples, block_size, seed=0):
    """Return list of (input_ids, labels) with the prompt masked to -1."""
    with open(os.path.join(DATA_DIR, "corpus.txt"), encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split(chat.EOT) if d.strip()]
    rng = random.Random(seed)
    rng.shuffle(docs)

    uid = tok.special_tokens[chat.USER]
    aid = tok.special_tokens[chat.ASSISTANT]
    eid = tok.special_tokens[chat.EOT]

    examples = []
    for story in docs:
        words = chat.pick_words(story, k=rng.choice([2, 3]), rng=rng)
        if not words:
            continue
        prompt_ids = [uid] + tok.encode_ordinary(chat.build_prompt(words)) + [aid]
        resp_ids = tok.encode_ordinary(story) + [eid]
        if len(prompt_ids) + len(resp_ids) > block_size:
            resp_ids = resp_ids[: block_size - len(prompt_ids)]
            if len(resp_ids) < 8:
                continue
        seq = prompt_ids + resp_ids
        # next-token targets, with prompt region masked
        x = seq[:-1]
        y = seq[1:]
        p = len(prompt_ids)              # first supervised target is seq[p]
        labels = [-1] * (p - 1) + y[p - 1:]
        examples.append((x, labels))
        if len(examples) >= n_examples:
            break
    return examples


def make_batch(examples, idx, pad_id, device):
    batch = [examples[i] for i in idx]
    maxlen = max(len(x) for x, _ in batch)
    X = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    Y = torch.full((len(batch), maxlen), -1, dtype=torch.long)
    for r, (x, y) in enumerate(batch):
        X[r, : len(x)] = torch.tensor(x)
        Y[r, : len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=os.path.join(OUT_DIR, "ckpt.pt"))
    p.add_argument("--out", default=os.path.join(OUT_DIR, "sft.pt"))
    p.add_argument("--n_examples", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else \
        torch.autocast(device_type="cpu", enabled=False)

    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    ckpt = torch.load(args.init, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    pad_id = tok.special_tokens[chat.EOT]

    examples = build_examples(tok, args.n_examples, cfg.block_size)
    print(f"SFT examples: {len(examples):,} | init {os.path.basename(args.init)}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.Random(epoch).shuffle(order)
        for s in range(0, len(order) - args.batch_size, args.batch_size):
            X, Y = make_batch(examples, order[s:s + args.batch_size], pad_id, device)
            with ctx:
                _, loss = model(X, Y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f} | {time.time()-t0:.0f}s")

    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, args.out)
    print(f"saved SFT model -> {args.out}")


if __name__ == "__main__":
    main()
