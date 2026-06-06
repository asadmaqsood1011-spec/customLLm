"""Direct Preference Optimization, implemented from the loss equation.

DPO (Rafailov et al., 2023) aligns the model to preference pairs without a
separate reward model or RL loop. For each (prompt, chosen, rejected) we push
the policy to raise the log-prob of `chosen` over `rejected`, *relative to a
frozen reference model* (the SFT checkpoint), so it doesn't drift far from it:

    pi_logratio  = logp_policy(chosen)    - logp_policy(rejected)
    ref_logratio = logp_ref(chosen)       - logp_ref(rejected)
    loss = -log_sigmoid( beta * (pi_logratio - ref_logratio) )

logp is summed over response tokens only (prompt masked). The implicit reward
is beta*(logp_policy - logp_ref); we track its chosen-minus-rejected margin and
the preference accuracy.

    python dpo.py            # out/sft.pt -> out/dpo.pt
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

import chat
from bpe import BPETokenizer
from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def build_seq(tok, prompt_text, response, block_size):
    uid = tok.special_tokens[chat.USER]
    aid = tok.special_tokens[chat.ASSISTANT]
    eid = tok.special_tokens[chat.EOT]
    prompt_ids = [uid] + tok.encode_ordinary(prompt_text) + [aid]
    resp_ids = tok.encode_ordinary(response) + [eid]
    seq = (prompt_ids + resp_ids)[:block_size]
    x, y = seq[:-1], seq[1:]
    p = len(prompt_ids)
    labels = [-1] * (p - 1) + y[p - 1:]
    return x, labels


def pad_batch(rows, pad_id, device):
    maxlen = max(len(x) for x, _ in rows)
    X = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    Y = torch.full((len(rows), maxlen), -1, dtype=torch.long)
    for r, (x, y) in enumerate(rows):
        X[r, : len(x)] = torch.tensor(x)
        Y[r, : len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)


def seq_logprob(model, X, Y, ctx):
    with ctx:
        logits = model(X)[0]
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = (Y != -1)
    tok_lp = logp.gather(-1, Y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=os.path.join(OUT_DIR, "sft.pt"))
    p.add_argument("--out", default=os.path.join(OUT_DIR, "dpo.pt"))
    p.add_argument("--pairs", default=os.path.join(DATA_DIR, "dpo_pairs.jsonl"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else \
        torch.autocast(device_type="cpu", enabled=False)

    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    ckpt = torch.load(args.init, map_location=device)
    cfg = GPTConfig(**ckpt["config"])

    policy = GPT(cfg).to(device)
    policy.load_state_dict(ckpt["model"])
    ref = GPT(cfg).to(device)
    ref.load_state_dict(ckpt["model"])
    for q in ref.parameters():
        q.requires_grad_(False)
    # eval mode on both: deterministic log-probs (no dropout noise)
    policy.eval()
    ref.eval()
    pad_id = tok.special_tokens[chat.EOT]

    pairs = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pairs.append((build_seq(tok, d["prompt"], d["chosen"], cfg.block_size),
                          build_seq(tok, d["prompt"], d["rejected"], cfg.block_size)))
    print(f"DPO pairs: {len(pairs):,} | beta={args.beta} lr={args.lr}")

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        random.Random(epoch).shuffle(order)
        for s in range(0, len(order) - args.batch_size, args.batch_size):
            idx = order[s:s + args.batch_size]
            Xc, Yc = pad_batch([pairs[i][0] for i in idx], pad_id, device)
            Xr, Yr = pad_batch([pairs[i][1] for i in idx], pad_id, device)

            lp_pol_c = seq_logprob(policy, Xc, Yc, ctx)
            lp_pol_r = seq_logprob(policy, Xr, Yr, ctx)
            with torch.no_grad():
                lp_ref_c = seq_logprob(ref, Xc, Yc, ctx)
                lp_ref_r = seq_logprob(ref, Xr, Yr, ctx)

            logits = args.beta * ((lp_pol_c - lp_pol_r) - (lp_ref_c - lp_ref_r))
            loss = -F.logsigmoid(logits).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

            step += 1
            if step % 50 == 0:
                with torch.no_grad():
                    r_c = args.beta * (lp_pol_c - lp_ref_c)
                    r_r = args.beta * (lp_pol_r - lp_ref_r)
                    margin = (r_c - r_r).mean().item()
                    acc = (r_c > r_r).float().mean().item()
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f} | "
                      f"reward margin {margin:+.4f} | pref acc {acc:.2f}")

    torch.save({"model": policy.state_dict(), "config": cfg.__dict__}, args.out)
    print(f"saved DPO model -> {args.out}")


if __name__ == "__main__":
    main()
