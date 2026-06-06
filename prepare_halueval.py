"""Download HaluEval (QA) and build labeled faithfulness examples.

HaluEval (Li et al., 2023) gives, per item: a knowledge passage, a question, a
correct answer, and a hallucinated answer. We turn each row into two labeled
(source, claim) pairs:

    (knowledge+question, right_answer)        -> 0  (supported / faithful)
    (knowledge+question, hallucinated_answer) -> 1  (hallucinated)

So the task is: given a source and a claim, did the claim stay faithful to the
source? Output: data/hallu_{train,val,test}.jsonl

    python prepare_halueval.py
"""

import json
import os
import random
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, "qa_data.json")
    if not os.path.exists(raw):
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(raw, "wb") as f:
            f.write(r.read())

    examples = []
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            source = f"Question: {d['question']}\nKnowledge: {d['knowledge']}"
            examples.append({"source": source, "claim": d["right_answer"], "label": 0})
            examples.append({"source": source, "claim": d["hallucinated_answer"], "label": 1})

    random.Random(0).shuffle(examples)
    n = len(examples)
    n_test = n_val = n // 10
    splits = {
        "test": examples[:n_test],
        "val": examples[n_test:n_test + n_val],
        "train": examples[n_test + n_val:],
    }
    for name, rows in splits.items():
        path = os.path.join(DATA_DIR, f"hallu_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in rows)
        print(f"{name:5}: {len(rows):>6,} examples ({pos/len(rows)*100:.0f}% hallucinated)")


if __name__ == "__main__":
    main()
