"""Audit faithfulness benchmarks for cheap shortcuts.

A faithfulness benchmark is supposed to test whether a claim is supported by its
source. If a trivial feature of the claim alone (its length, how many words are
missing from the source, how many negations it has) predicts the label, then a
model can score well without doing the actual task. That is a benchmark
artifact, and it inflates reported numbers.

This module scores a set of model-free cues on any split and reports the AUROC
of each. An AUROC near 0.5 means the cue is useless (good, no shortcut). An
AUROC far from 0.5 means the label leaks through that cue (a shortcut a model
can exploit).

Run it across whatever benchmarks are present in data/:

    python shortcuts.py                  # audit every benchmark found
    python shortcuts.py --bench qa summ  # audit specific ones
"""

import argparse
import os
import re

import numpy as np

import hallu_data as hd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Benchmarks we know how to load, in display order. A benchmark "x" is present
# when data/hallu_x_test.jsonl exists (built by prepare_benchmarks.py).
KNOWN = ["qa", "summ", "dial", "fever"]


def auroc(scores, labels):
    """Probability a random positive outranks a random negative (Mann-Whitney)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


_WORD = re.compile(r"[a-z]+")
_NEG = re.compile(r"\b(no|not|never|none|cannot|n't|without|neither|nor)\b")


def _words(text):
    return _WORD.findall(text.lower())


# --- cues: each maps a row -> a scalar, higher meaning "looks hallucinated" ---

def cue_claim_chars(r):
    return len(r["claim"])


def cue_claim_words(r):
    return len(_words(r["claim"]))


def cue_source_chars(r):
    return len(r["source"])


def cue_novel_word_frac(r):
    """Fraction of claim words absent from the source (lexical-overlap shortcut)."""
    src = set(_words(r["source"]))
    cw = _words(r["claim"])
    if not cw:
        return 0.0
    return 1.0 - sum(w in src for w in cw) / len(cw)


def cue_len_ratio(r):
    return len(r["claim"]) / (len(r["source"]) + 1)


def cue_negations(r):
    return len(_NEG.findall(r["claim"].lower()))


CUES = {
    "claim length (chars)": cue_claim_chars,
    "claim length (words)": cue_claim_words,
    "source length (chars)": cue_source_chars,
    "novel-word fraction": cue_novel_word_frac,
    "claim/source ratio": cue_len_ratio,
    "negation count": cue_negations,
}


def audit_rows(rows):
    """Return {cue_name: auroc} for one set of labeled rows."""
    labels = [r["label"] for r in rows]
    return {name: auroc([fn(r) for r in rows], labels) for name, fn in CUES.items()}


def gameability(aurocs):
    """How exploitable the benchmark is = how far the best cue strays from 0.5."""
    return max(abs(a - 0.5) for a in aurocs.values()) + 0.5


def present_benchmarks():
    found = []
    for b in KNOWN:
        if os.path.exists(os.path.join(DATA_DIR, f"hallu_{b}_test.jsonl")):
            found.append(b)
    # the original QA build used unprefixed files (data/hallu_test.jsonl)
    if "qa" not in found and os.path.exists(os.path.join(DATA_DIR, "hallu_test.jsonl")):
        found.append("qa")
    return found


def _load(bench, split):
    name = split if bench == "qa" and not os.path.exists(
        os.path.join(DATA_DIR, f"hallu_qa_{split}.jsonl")
    ) else f"{bench}_{split}"
    return hd.load_split(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="*", default=None, help="benchmarks to audit")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    benches = args.bench or present_benchmarks()
    if not benches:
        print("no benchmarks found in data/. run prepare_benchmarks.py first.")
        return

    results = {}
    for b in benches:
        rows = _load(b, args.split)
        results[b] = (len(rows), audit_rows(rows))

    cue_names = list(CUES)
    width = max(len(c) for c in cue_names) + 2
    header = f"{'cue'.ljust(width)}" + "".join(f"{b:>12}" for b in benches)
    print("\nShortcut audit  (AUROC of a model-free cue; 0.50 = no shortcut)\n")
    print(header)
    print("-" * len(header))
    for c in cue_names:
        row = c.ljust(width) + "".join(f"{results[b][1][c]:>12.3f}" for b in benches)
        print(row)
    print("-" * len(header))
    print("gameability".ljust(width)
          + "".join(f"{gameability(results[b][1]):>12.3f}" for b in benches))
    print("\n(gameability = best single-cue AUROC; 0.50 ideal, 1.00 fully gamed)")
    print("examples:" + "".join(f"  {b}={results[b][0]:,}" for b in benches))


if __name__ == "__main__":
    main()
