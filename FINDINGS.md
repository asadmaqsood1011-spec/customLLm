# Are faithfulness benchmarks measuring faithfulness?

A faithfulness benchmark should test whether a claim is supported by its source. If a trivial property of the claim alone predicts the label, a model can score well without doing the task. Below, each number is the AUROC of a single model-free cue. 0.50 means the cue is useless (good). Far from 0.50 means the benchmark leaks the answer through that cue.

Benchmarks audited (all source-grounded subsets of HaluEval): `qa` (2,000 test), `summ` (2,000 test), `dial` (2,000 test).

## Raw benchmarks

| cue | qa | summ | dial |
|---|---|---|---|
| claim length (chars) | 0.976 | 0.767 | 0.725 |
| claim length (words) | 0.978 | 0.768 | 0.709 |
| source length (chars) | 0.489 | 0.508 | 0.491 |
| novel-word fraction | 0.848 | 0.724 | 0.483 |
| claim/source ratio | 0.966 | 0.681 | 0.725 |
| negation count | 0.540 | 0.546 | 0.563 |
| **best cue (gameability)** | **0.978** | **0.768** | **0.725** |

Every benchmark is gameable by claim length. The worst is `qa`, where **claim length (words) alone scores 0.978 AUROC**: the hallucinated answers are simply longer. A detector can learn to count characters and look good.

## After matching the claim-length distribution

| cue | qa | summ | dial |
|---|---|---|---|
| claim length (chars) | 0.731 | 0.502 | 0.500 |
| claim length (words) | 0.780 | 0.506 | 0.483 |
| source length (chars) | 0.386 | 0.412 | 0.443 |
| novel-word fraction | 0.871 | 0.760 | 0.491 |
| claim/source ratio | 0.714 | 0.583 | 0.531 |
| negation count | 0.557 | 0.513 | 0.501 |
| **best cue (gameability)** | **0.871** | **0.760** | **0.557** |

Length control fully neutralizes the cue on summarization and dialogue (AUROC ~0.50). On QA it drops sharply but a residual remains, and a second cue appears: novel-word fraction (claim words missing from the source). That one is partly legitimate, since missing source words genuinely signal unfaithfulness, so we neutralize the clearly-spurious length cue and report what the detector does against the rest.

## Does the detector beat the shortcuts?

HalluGuard (12M params, from scratch) trained and tested on the length-controlled splits, next to the best trivial cue that survives control. The detector only earns credit where it clears the cues.

| benchmark | detector AUROC | detector acc | best surviving cue |
|---|---|---|---|
| qa | 0.977 | 0.948 | novel-word fraction 0.871 |
| summ | 1.000 | 1.000 | novel-word fraction 0.760 |
| dial | 0.875 | 0.779 | source length (chars) 0.443 |

The cleanest test is `dial`, where every trivial cue is already ~0.50, so there is no shortcut left to ride. The detector still scores 0.875 AUROC there, which is the honest evidence that it learned real faithfulness signal rather than an artifact. Where cues survive (QA length and lexical overlap), the high scores are partly those cues, and we say so.

## Takeaway

Reported accuracy on these benchmarks is inflated by claim length. Any honest result should either control for it or report the length-cue baseline alongside the model. The length-controlled splits are built by `make_controlled.py` and the detector is evaluated on them in `eval_cls.py`.
