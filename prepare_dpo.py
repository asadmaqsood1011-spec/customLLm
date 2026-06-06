"""Build DPO preference pairs by sampling the SFT model and ranking with a
heuristic reward.

For each prompt we draw two completions from the SFT model and score them with
a cheap, transparent reward:
    reward = word_hit_rate + 0.3*distinct2 + length_bonus
The higher-scoring completion is `chosen`, the other `rejected`. This is a
reward *proxy* (stated honestly in the report) standing in for a human/learned
reward model — the DPO machinery is identical either way.

    python prepare_dpo.py            # -> data/dpo_pairs.jsonl
"""

import argparse
import json
import os
import random

import torch

import chat
from infer import load, chat_sample_k

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def reward(text, words):
    hit = chat.word_hit_rate(text, words)
    div = chat.distinct2(text)
    nwords = len(text.split())
    length_bonus = min(nwords, 60) / 60 * 0.2     # prefer non-trivial, capped
    return hit + 0.3 * div + length_bonus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default=os.path.join(OUT_DIR, "sft.pt"))
    p.add_argument("--n_prompts", type=int, default=1500)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--out", default=os.path.join(DATA_DIR, "dpo_pairs.jsonl"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model, _ = load(args.sft, device)

    with open(os.path.join(DATA_DIR, "corpus.txt"), encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split(chat.EOT) if d.strip()]
    rng = random.Random(123)
    rng.shuffle(docs)

    n_pairs = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for story in docs:
            if n_pairs >= args.n_prompts:
                break
            words = chat.pick_words(story, k=rng.choice([2, 3]), rng=rng)
            if not words:
                continue
            prompt = chat.build_prompt(words)
            a, b = chat_sample_k(model, tok, prompt, 2, device,
                                 max_new_tokens=args.max_new_tokens, temperature=1.0)
            ra, rb = reward(a, words), reward(b, words)
            if abs(ra - rb) < 1e-3 or not a or not b:
                continue
            chosen, rejected = (a, b) if ra > rb else (b, a)
            fout.write(json.dumps({"prompt": prompt, "chosen": chosen,
                                   "rejected": rejected, "words": words}) + "\n")
            n_pairs += 1
            if n_pairs % 200 == 0:
                print(f"{n_pairs} pairs ...")

    print(f"wrote {n_pairs} preference pairs -> {args.out}")


if __name__ == "__main__":
    main()
