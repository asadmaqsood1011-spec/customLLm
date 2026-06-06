"""Download TinyShakespeare, train the BPE tokenizer, and tokenize the corpus.

Run once before training:  python data.py
Produces in data/: input.txt, tokenizer.json, train.bin, val.bin
"""

import os
import urllib.request

import numpy as np

from bpe import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
VOCAB_SIZE = 4096


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    txt_path = os.path.join(DATA_DIR, "input.txt")

    if not os.path.exists(txt_path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, txt_path)
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} chars")

    print(f"training BPE to vocab_size={VOCAB_SIZE} ...")
    tok = BPETokenizer()
    tok.train(text, VOCAB_SIZE, verbose=True)
    tok.save(os.path.join(DATA_DIR, "tokenizer.json"))
    print(f"tokenizer vocab: {tok.vocab_size}")

    print("encoding corpus ...")
    ids = tok.encode(text)
    print(f"tokens: {len(ids):,}  (compression {len(text)/len(ids):.2f}x)")

    n = int(0.9 * len(ids))
    train_ids = np.array(ids[:n], dtype=np.uint16)
    val_ids = np.array(ids[n:], dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"train {len(train_ids):,} tokens | val {len(val_ids):,} tokens")


if __name__ == "__main__":
    main()
