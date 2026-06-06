"""Byte-level BPE tokenizer, written from scratch.

Same core algorithm GPT-2/GPT-4 use: start from raw UTF-8 bytes (256 base
tokens, so every string is always encodable), then repeatedly merge the most
frequent adjacent token pair into a new token until we hit the target vocab.

Two details that make it fast and faithful to real BPE:
  - text is first split into word-like chunks with a regex (a simplified GPT-2
    pattern), so merges never cross word boundaries;
  - identical chunks are de-duplicated and counted, so each training pass scans
    the set of *unique* words rather than the whole corpus.
"""

import json
import re
from collections import Counter

# Simplified GPT-2 split pattern (ASCII-oriented; fine for English text).
# Keeps leading spaces attached to words, groups digits and punctuation runs.
PAT = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def get_stats(ids, counts=None, weight=1):
    """Tally adjacent pairs into `counts`, scaled by `weight`."""
    counts = {} if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + weight
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
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
        self.merges = {}        # (a, b) -> new_id
        self.vocab = {}         # id -> bytes
        self._cache = {}        # chunk str -> encoded ids

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # de-duplicate word chunks; train over unique words weighted by count
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
            if verbose and (n + 1) % 200 == 0:
                print(f"merge {n+1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}) x{stats[pair]}")

        self.merges = merges
        self.vocab = vocab
        self._cache = {}

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # merge the pair we learned earliest (lowest new_id)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in PAT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        data = {"merges": [[a, b, nid] for (a, b), nid in self.merges.items()]}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for a, b, nid in data["merges"]:
            merges[(a, b)] = nid
            vocab[nid] = vocab[a] + vocab[b]
        self.merges = merges
        self.vocab = vocab
        self._cache = {}
        return self
