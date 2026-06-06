"""Sample text from a trained checkpoint.

    python generate.py --prompt "ROMEO:" --max_new_tokens 500
"""

import argparse
import os

import torch

from bpe import BPETokenizer
from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded ckpt @ iter {ckpt['iter']} (val loss {ckpt['val_loss']:.4f})\n")

    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
