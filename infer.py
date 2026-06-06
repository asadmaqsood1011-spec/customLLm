"""Load a checkpoint and run chat-formatted generation. Shared by DPO + eval."""

import os

import torch

import chat
from bpe import BPETokenizer
from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load(path, device):
    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return tok, model, cfg


@torch.no_grad()
def chat_generate(model, tok, prompt_text, device, max_new_tokens=200,
                  temperature=0.8, top_k=100):
    """Return the assistant's response text for one user instruction."""
    uid = tok.special_tokens[chat.USER]
    aid = tok.special_tokens[chat.ASSISTANT]
    eid = tok.special_tokens[chat.EOT]
    ids = [uid] + tok.encode_ordinary(prompt_text) + [aid]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, eos_id=eid)[0].tolist()
    resp = out[len(ids):]
    if eid in resp:
        resp = resp[: resp.index(eid)]
    return tok.decode(resp).strip()


@torch.no_grad()
def chat_sample_k(model, tok, prompt_text, k, device, max_new_tokens=96,
                  temperature=1.0, top_k=100):
    """Sample k diverse responses for one prompt in a single batched pass."""
    uid = tok.special_tokens[chat.USER]
    aid = tok.special_tokens[chat.ASSISTANT]
    eid = tok.special_tokens[chat.EOT]
    ids = [uid] + tok.encode_ordinary(prompt_text) + [aid]
    idx = torch.tensor([ids], dtype=torch.long, device=device).repeat(k, 1)
    out = model.generate(idx, max_new_tokens, temperature, top_k)  # no early stop (batch)
    resps = []
    for row in out.tolist():
        r = row[len(ids):]
        if eid in r:
            r = r[: r.index(eid)]
        resps.append(tok.decode(r).strip())
    return resps
