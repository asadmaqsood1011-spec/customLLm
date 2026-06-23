"""Download several source-grounded faithfulness benchmarks into one format.

Each benchmark gives a source and two claims about it, one faithful and one
hallucinated. We normalize every benchmark to the same labeled shape so the
shortcut audit and the detector can run on all of them unchanged:

    {"source": <grounding text>, "claim": <answer/summary/response>, "label": 0|1}
    label 0 = supported by the source, 1 = hallucinated

All three subsets come from HaluEval (Li et al., 2023), which is the cleanest
public set of paired faithful/hallucinated claims with an explicit source. We
deliberately use only source-grounded tasks (faithfulness), not world-knowledge
factuality, because a 12M detector can check "supported by THIS text" but cannot
know all world facts.

    python prepare_benchmarks.py            # all benchmarks
    python prepare_benchmarks.py --bench qa summ
"""

import argparse
import json
import os
import random
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"


def _download(fname):
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, fname)
    if not os.path.exists(raw):
        url = f"{BASE}/{fname}"
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(raw, "wb") as f:
            f.write(r.read())
    return raw


def _rows(raw):
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- per-benchmark extractors: raw record -> (source, faithful, hallucinated) --

def extract_qa(d):
    source = f"Question: {d['question']}\nKnowledge: {d['knowledge']}"
    return source, d["right_answer"], d["hallucinated_answer"]


def extract_summ(d):
    return d["document"], d["right_summary"], d["hallucinated_summary"]


def extract_dial(d):
    source = f"Knowledge: {d['knowledge']}\nDialogue: {d['dialogue_history']}"
    return source, d["right_response"], d["hallucinated_response"]


BENCHMARKS = {
    "qa":   ("qa_data.json",            extract_qa),
    "summ": ("summarization_data.json", extract_summ),
    "dial": ("dialogue_data.json",      extract_dial),
}


def build(bench):
    fname, extract = BENCHMARKS[bench]
    raw = _download(fname)
    examples = []
    for d in _rows(raw):
        source, ok, bad = extract(d)
        examples.append({"source": source, "claim": ok, "label": 0})
        examples.append({"source": source, "claim": bad, "label": 1})

    random.Random(0).shuffle(examples)
    n = len(examples)
    n_test = n_val = n // 10
    splits = {
        "test": examples[:n_test],
        "val": examples[n_test:n_test + n_val],
        "train": examples[n_test + n_val:],
    }
    for split, rows in splits.items():
        path = os.path.join(DATA_DIR, f"hallu_{bench}_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"{bench:5}: {n:>7,} examples -> "
          f"{len(splits['train']):,}/{len(splits['val']):,}/{len(splits['test']):,} "
          f"train/val/test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="*", default=list(BENCHMARKS))
    args = p.parse_args()
    for b in args.bench:
        build(b)


if __name__ == "__main__":
    main()
