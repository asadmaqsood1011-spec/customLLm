"""Build length-controlled versions of the faithfulness splits.

The audit (shortcuts.py) shows every benchmark here leaks the label through
claim length: the hallucinated answers are simply longer. A detector can ride
that cue instead of judging faithfulness. We remove it by matching the
claim-length distribution between the two classes: bucket by character length
and, in each bucket, downsample the larger class to the size of the smaller.
After this a length cue is worthless (~0.5 AUROC), so any signal a model keeps
is real faithfulness signal.

    python make_controlled.py                 # control every benchmark found
    python make_controlled.py --bench qa      # just one

For benchmark "qa" this writes data/hallu_qa_ctrl_{train,val,test}.jsonl.
"""

import argparse
import json
import os
import random

import hallu_data as hd
from shortcuts import present_benchmarks, _load

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BUCKET = 20   # characters per length bucket


def control(rows, rng):
    buckets = {}
    for r in rows:
        b = len(r["claim"]) // BUCKET
        buckets.setdefault(b, {0: [], 1: []})[r["label"]].append(r)
    out = []
    for byc in buckets.values():
        k = min(len(byc[0]), len(byc[1]))
        if k == 0:
            continue
        out += rng.sample(byc[0], k) + rng.sample(byc[1], k)
    rng.shuffle(out)
    return out


def build(bench, rng):
    for split in ("train", "val", "test"):
        rows = _load(bench, split)
        ctrl = control(rows, rng)
        path = os.path.join(DATA_DIR, f"hallu_{bench}_ctrl_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in ctrl:
                f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in ctrl)
        print(f"{bench}/{split:5}: {len(rows):>6,} -> {len(ctrl):>6,} length-matched "
              f"({pos/max(len(ctrl),1)*100:.0f}% hallucinated)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="*", default=None)
    args = p.parse_args()
    benches = args.bench or present_benchmarks()
    if not benches:
        print("no benchmarks found in data/. run prepare_benchmarks.py first.")
        return
    rng = random.Random(0)
    for b in benches:
        build(b, rng)


if __name__ == "__main__":
    main()
