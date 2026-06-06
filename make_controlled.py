"""Build a length-controlled version of the HaluEval splits.

The raw benchmark is gameable: hallucinated answers tend to be longer, so a
claim-length heuristic scores ~0.98 AUROC. We remove that shortcut by matching
the claim-length distribution between the two classes (bucketed by character
length, downsampled to the smaller class per bucket). On the resulting data a
length cue is worthless (~0.5 AUROC), so any signal the model shows is real.

    python make_controlled.py        # -> data/hallu_ctrl_{train,val,test}.jsonl
"""

import json
import os
import random

import hallu_data as hd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BUCKET = 20   # characters


def control(rows, rng):
    buckets = {}
    for r in rows:
        b = len(r["claim"]) // BUCKET
        buckets.setdefault(b, {0: [], 1: []})[r["label"]].append(r)
    out = []
    for b, byc in buckets.items():
        k = min(len(byc[0]), len(byc[1]))
        if k == 0:
            continue
        out += rng.sample(byc[0], k) + rng.sample(byc[1], k)
    rng.shuffle(out)
    return out


def main():
    rng = random.Random(0)
    for split in ("train", "val", "test"):
        rows = hd.load_split(split)
        ctrl = control(rows, rng)
        path = os.path.join(DATA_DIR, f"hallu_ctrl_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in ctrl:
                f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in ctrl)
        print(f"{split:5}: {len(rows):>6,} -> {len(ctrl):>6,} length-matched "
              f"({pos/len(ctrl)*100:.0f}% hallucinated)")


if __name__ == "__main__":
    main()
