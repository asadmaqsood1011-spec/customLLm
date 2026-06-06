"""Evaluate base / SFT / DPO models on held-out instruction prompts.

Reports, per stage:
  - word hit-rate : fraction of requested words the response actually uses
                    (instruction-following — the thing SFT should fix)
  - distinct-2    : unique-bigram fraction (anti-repetition — DPO's reward proxy)
  - avg words     : response length

Prompts are drawn with a seed unused by SFT/DPO training, so this is held out.

    python eval.py
"""

import argparse
import os
import random

import torch

import chat
from infer import load, chat_generate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def held_out_prompts(n, seed=999):
    with open(os.path.join(DATA_DIR, "corpus.txt"), encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split(chat.EOT) if d.strip()]
    rng = random.Random(seed)
    rng.shuffle(docs)
    prompts = []
    for story in docs:
        words = chat.pick_words(story, k=rng.choice([2, 3]), rng=rng)
        if words:
            prompts.append(words)
        if len(prompts) >= n:
            break
    return prompts


def evaluate(ckpt, prompts, device, seed=0):
    tok, model, _ = load(ckpt, device)
    hits, divs, lens = [], [], []
    for words in prompts:
        torch.manual_seed(seed)        # same sampling noise across models
        resp = chat_generate(model, tok, chat.build_prompt(words), device,
                             max_new_tokens=160, temperature=0.7, top_k=100)
        hits.append(chat.word_hit_rate(resp, words))
        divs.append(chat.distinct2(resp))
        lens.append(len(resp.split()))
    n = len(prompts)
    return sum(hits) / n, sum(divs) / n, sum(lens) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = held_out_prompts(args.n)
    stages = [("base", "ckpt.pt"), ("SFT", "sft.pt"), ("DPO", "dpo.pt")]

    print(f"held-out prompts: {len(prompts)}\n")
    print(f"{'stage':<6} {'word hit-rate':>14} {'distinct-2':>12} {'avg words':>11}")
    print("-" * 46)
    for name, fn in stages:
        path = os.path.join(OUT_DIR, fn)
        if not os.path.exists(path):
            continue
        hit, div, ln = evaluate(path, prompts, device)
        print(f"{name:<6} {hit:>14.3f} {div:>12.3f} {ln:>11.1f}")


if __name__ == "__main__":
    main()
