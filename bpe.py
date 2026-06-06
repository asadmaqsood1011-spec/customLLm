"""Byte-level BPE tokenizer, written from scratch.

Same core algorithm GPT-2/GPT-4 use: start from raw UTF-8 bytes (256 base
tokens, so every string is always encodable), then repeatedly merge the most
frequent adjacent token pair into a new token until we hit the target vocab.

Two details that make it fast and faithful to real BPE:
  - text is first split into word-like chunks with a regex (a simplified GPT-2
    pattern), so merges never cross word boundaries;
  - identical chunks are de-duplicated and counted, so each training pass scans
    the set of *unique* words rather than the whole corpus.

Special tokens (e.g. <|endoftext|>, <|user|>, <|assistant|>) are registered
after training and given ids above the BPE range; encode() can splice them in.
"""

import json
import re
from collections import Counter

# Simplified GPT-2 split pattern (ASCII-oriented; fine for English text).
PAT = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def get_stats(ids, counts=None, weight=1):
    counts = {} if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + weight
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}            # (a, b) -> new_id
        self.vocab = {}             # id -> bytes
        self.special_tokens = {}    # str -> id
        self._special_inv = {}      # id -> str
        self._cache = {}

    # ---- training ----------------------------------------------------------
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        word_counts = Counter(PAT.findall(text))
        words = [list(w.encode("utf-8")) for w in word_counts]
        weights = list(word_counts.values())

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for n in range(num_merges):
            stats = {}
            for ids, w in zip(words, weights):
                get_stats(ids, stats, w)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + n
            words = [merge(ids, pair, new_id) for ids in words]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (n + 1) % 500 == 0:
                print(f"merge {n+1}/{num_merges}: {vocab[new_id]!r} x{stats[pair]}")
        self.merges = merges
        self.vocab = vocab
        self._cache = {}

    def register_special(self, tokens):
        """Assign ids to special-token strings, above the BPE vocab."""
        nxt = len(self.vocab)
        for t in tokens:
            if t not in self.special_tokens:
                self.special_tokens[t] = nxt
                self._special_inv[nxt] = t
                nxt += 1

    # ---- encoding / decoding ----------------------------------------------
    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        ids = []
        for chunk in PAT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text, allow_special=True):
        """Encode text, splicing in any registered special tokens found verbatim."""
        if not allow_special or not self.special_tokens:
            return self.encode_ordinary(text)
        # split while keeping the special-token delimiters
        pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self._special_inv:
                parts.append(self._special_inv[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        data = {
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for a, b, nid in data["merges"]:
            merges[(a, b)] = nid
            vocab[nid] = vocab[a] + vocab[b]
        self.merges, self.vocab = merges, vocab
        self.special_tokens = data.get("special_tokens", {})
        self._special_inv = {v: k for k, v in self.special_tokens.items()}
        self._cache = {}
        return self
