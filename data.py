"""Prepare the pretraining corpus: download, train BPE, tokenize to .bin.

Default corpus is TinyStories (Eldan & Li, 2023) — short synthetic stories in
simple English, designed so that small models can learn to produce *coherent*
text. Falls back to TinyShakespeare if the download fails.

Run once:  python data.py
Produces in data/: tokenizer.json, train.bin, val.bin
"""

import os
import urllib.request

import numpy as np

from bpe import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EOT = "<|endoftext|>"
SPECIALS = [EOT, "<|user|>", "<|assistant|>"]

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

CORPUS_BYTES = 60 * 1024 * 1024     # cap how much we download
BPE_TRAIN_BYTES = 8 * 1024 * 1024   # train tokenizer on a prefix (fast)
VOCAB_SIZE = 4096


def download_capped(url, max_bytes):
    """Stream-download up to max_bytes and return decoded text."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read(max_bytes)
    return raw.decode("utf-8", errors="ignore")


def load_corpus():
    cache = os.path.join(DATA_DIR, "corpus.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return f.read()
    try:
        print("downloading TinyStories (capped) ...")
        text = download_capped(TINYSTORIES_URL, CORPUS_BYTES)
        # drop the last (likely truncated) story
        text = text.rsplit(EOT, 1)[0] + EOT
    except Exception as e:
        print(f"TinyStories download failed ({e}); falling back to TinyShakespeare")
        text = download_capped(SHAKESPEARE_URL, CORPUS_BYTES) + "\n" + EOT
    with open(cache, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    text = load_corpus()
    docs = [d.strip() for d in text.split(EOT) if d.strip()]
    print(f"corpus: {len(text):,} chars | {len(docs):,} documents")

    print(f"training BPE (vocab {VOCAB_SIZE}) on a {BPE_TRAIN_BYTES//1024//1024}MB prefix ...")
    tok = BPETokenizer()
    tok.train(text[:BPE_TRAIN_BYTES], VOCAB_SIZE, verbose=True)
    tok.register_special(SPECIALS)
    tok.save(os.path.join(DATA_DIR, "tokenizer.json"))
    eot_id = tok.special_tokens[EOT]
    print(f"vocab (incl. specials): {tok.vocab_size} | <|endoftext|>={eot_id}")

    print("encoding corpus (one <|endoftext|> between documents) ...")
    ids = []
    for d in docs:
        ids.extend(tok.encode_ordinary(d))
        ids.append(eot_id)
    print(f"tokens: {len(ids):,}  (compression {len(text)/len(ids):.2f}x)")

    n = int(0.9 * len(ids))
    np.array(ids[:n], dtype=np.uint16).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.array(ids[n:], dtype=np.uint16).tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"train {n:,} tokens | val {len(ids)-n:,} tokens")


if __name__ == "__main__":
    main()
