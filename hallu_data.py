"""Encoding + batching for the faithfulness classifier. Shared by train/eval."""

import json
import os

import torch

import chat

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_split(name):
    rows = []
    with open(os.path.join(DATA_DIR, f"hallu_{name}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def encode_example(tok, source, claim, block_size):
    """<|user|> source <|assistant|> claim, truncated to fit (claim kept)."""
    uid = tok.special_tokens[chat.USER]
    aid = tok.special_tokens[chat.ASSISTANT]
    claim_ids = tok.encode_ordinary(claim)
    if len(claim_ids) > block_size - 10:
        claim_ids = claim_ids[: block_size - 10]
    budget = block_size - 2 - len(claim_ids)
    src_ids = tok.encode_ordinary(source)[:budget]
    return [uid] + src_ids + [aid] + claim_ids


def make_batch(tok, rows, block_size, pad_id, device):
    seqs = [encode_example(tok, r["source"], r["claim"], block_size) for r in rows]
    maxlen = max(len(s) for s in seqs)
    X = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    for i, s in enumerate(seqs):
        X[i, : len(s)] = torch.tensor(s)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return X.to(device), lengths.to(device), labels.to(device)
