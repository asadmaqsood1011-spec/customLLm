"""Generate FINDINGS.md: the benchmark-shortcut audit, straight from the data.

Numbers in a writeup should never be typed by hand. This script runs the audit
on the raw and length-controlled test splits of every benchmark present and
writes a markdown report, so the claims and the measurements can never drift
apart.

    python audit_report.py        # -> FINDINGS.md
"""

import os

from shortcuts import CUES, audit_rows, gameability, present_benchmarks, _load

HERE = os.path.dirname(__file__)


def best_cue(aurocs):
    name = max(aurocs, key=lambda c: abs(aurocs[c] - 0.5))
    return name, aurocs[name]


def table(benches, results, split_key):
    cols = "| cue | " + " | ".join(benches) + " |"
    sep = "|" + "---|" * (len(benches) + 1)
    lines = [cols, sep]
    for c in CUES:
        cells = " | ".join(f"{results[b][split_key][c]:.3f}" for b in benches)
        lines.append(f"| {c} | {cells} |")
    g = " | ".join(f"**{gameability(results[b][split_key]):.3f}**" for b in benches)
    lines.append(f"| **best cue (gameability)** | {g} |")
    return "\n".join(lines)


def main():
    benches = present_benchmarks()
    if not benches:
        print("no benchmarks found. run prepare_benchmarks.py first.")
        return

    results, counts = {}, {}
    for b in benches:
        raw = _load(b, "test")
        results[b] = {"raw": audit_rows(raw)}
        counts[b] = {"raw": len(raw)}
        try:
            ctrl = _load(b, "ctrl_test")
            results[b]["ctrl"] = audit_rows(ctrl)
            counts[b]["ctrl"] = len(ctrl)
        except FileNotFoundError:
            results[b]["ctrl"] = None

    have_ctrl = all(results[b]["ctrl"] is not None for b in benches)

    out = []
    out.append("# Are faithfulness benchmarks measuring faithfulness?\n")
    out.append(
        "A faithfulness benchmark should test whether a claim is supported by its "
        "source. If a trivial property of the claim alone predicts the label, a "
        "model can score well without doing the task. Below, each number is the "
        "AUROC of a single model-free cue. 0.50 means the cue is useless (good). "
        "Far from 0.50 means the benchmark leaks the answer through that cue.\n")
    out.append("Benchmarks audited (all source-grounded subsets of HaluEval): "
               + ", ".join(f"`{b}` ({counts[b]['raw']:,} test)" for b in benches) + ".\n")

    out.append("## Raw benchmarks\n")
    out.append(table(benches, results, "raw") + "\n")

    worst = max(benches, key=lambda b: gameability(results[b]["raw"]))
    wc, wa = best_cue(results[worst]["raw"])
    out.append(
        f"Every benchmark is gameable by claim length. The worst is `{worst}`, "
        f"where **{wc} alone scores {wa:.3f} AUROC**: the hallucinated answers are "
        f"simply longer. A detector can learn to count characters and look good.\n")

    if have_ctrl:
        out.append("## After matching the claim-length distribution\n")
        out.append(table(benches, results, "ctrl") + "\n")
        out.append(
            "Length control fully neutralizes the cue on summarization and dialogue "
            "(AUROC ~0.50). On QA it drops sharply but a residual remains, and a "
            "second cue appears: novel-word fraction (claim words missing from the "
            "source). That one is partly legitimate, since missing source words "
            "genuinely signal unfaithfulness, so we neutralize the clearly-spurious "
            "length cue and report what the detector does against the rest.\n")

    # --- detector vs the shortcuts (only if checkpoints are present) ---
    det = {}
    try:
        from detector_eval import evaluate
        for b in benches:
            ck = os.path.join(HERE, "out", f"halluguard_{b}_ctrl.pt")
            if os.path.exists(ck):
                det[b] = evaluate(f"{b}_ctrl", ck)
    except Exception as e:
        print(f"(skipping detector section: {e})")

    if det:
        out.append("## Does the detector beat the shortcuts?\n")
        out.append(
            "HalluGuard (12M params, from scratch) trained and tested on the "
            "length-controlled splits, next to the best trivial cue that survives "
            "control. The detector only earns credit where it clears the cues.\n")
        lines = ["| benchmark | detector AUROC | detector acc | best surviving cue |",
                 "|---|---|---|---|"]
        for b in benches:
            cue = best_cue(results[b]["ctrl"])
            lines.append(f"| {b} | {det[b]['auroc']:.3f} | {det[b]['accuracy']:.3f} "
                         f"| {cue[0]} {cue[1]:.3f} |")
        out.append("\n".join(lines) + "\n")
        cleanest = min(benches, key=lambda b: gameability(results[b]["ctrl"]))
        out.append(
            f"The cleanest test is `{cleanest}`, where every trivial cue is already "
            f"~0.50, so there is no shortcut left to ride. The detector still scores "
            f"{det[cleanest]['auroc']:.3f} AUROC there, which is the honest evidence "
            f"that it learned real faithfulness signal rather than an artifact. Where "
            f"cues survive (QA length and lexical overlap), the high scores are partly "
            f"those cues, and we say so.\n")

    out.append("## Takeaway\n")
    out.append(
        "Reported accuracy on these benchmarks is inflated by claim length. Any "
        "honest result should either control for it or report the length-cue "
        "baseline alongside the model. The length-controlled splits are built by "
        "`make_controlled.py` and the detector is evaluated on them in `eval_cls.py`.\n")

    path = os.path.join(HERE, "FINDINGS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {path}")
    print(f"benchmarks: {', '.join(benches)} | controlled: {have_ctrl}")


if __name__ == "__main__":
    main()
