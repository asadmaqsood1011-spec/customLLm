"""Shared chat format + the instruction task, used by SFT, DPO and eval.

Task: given a few required words, write a short story that uses them. It's
learnable by a tiny model and trivially *measurable* — we can check whether a
generation actually contains the requested words. That measurability is the
whole point: it turns "did alignment work?" into a number.
"""

import random
import re

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|endoftext|>"
SPECIALS = [EOT, USER, ASSISTANT]

STOP = set(
    "the and a to of was were is are it he she they his her them you i in on at "
    "for with that this then there was had has have will would could said one day "
    "very so but not as be by an or from".split()
)


def pick_words(story, k=3, rng=random):
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", story)]
    cand = [w for w in dict.fromkeys(words) if len(w) >= 4 and w not in STOP]
    if not cand:
        return None
    return rng.sample(cand, min(k, len(cand)))


def build_prompt(words):
    return f"Write a short story using these words: {', '.join(words)}."


def contains_all(text, words):
    low = text.lower()
    return all(w in low for w in words)


def word_hit_rate(text, words):
    if not words:
        return 0.0
    low = text.lower()
    return sum(w in low for w in words) / len(words)


def distinct2(text):
    """distinct-2: fraction of unique bigrams (repetition proxy; higher=better)."""
    toks = re.findall(r"[A-Za-z]+", text.lower())
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return len(set(bigrams)) / len(bigrams)
