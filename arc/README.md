# ARC at laptop scale: learning a rule from a few examples

Experiment A of the token-free roadmap: can a small "settling" core learn a NEW grid-transformation rule from
the 2-5 examples a task provides? ARC-AGI-1 (github.com/fchollet/ARC-AGI, Apache-2.0), downloaded into
`data/ARC-AGI` (not committed). CPU only.

## Split (by rule, not by hand)

| Set | Source | Rule | Tasks |
|---|---|---|---|
| dev | ARC training tasks | output keeps the input's size; grids <= 15x15 | 173 |
| held-out | ARC evaluation tasks | same rule | 114 |

Methods are designed on dev and frozen before a single held-out run. A task counts only if every test output
is exactly right; 2 attempts allowed (as in ARC Prize).

## Results (2026-10-02)

| Method | Dev (173) | Held-out (114) |
|---|---|---|
| Copy the input | 0% | 0% |
| Transform search (rotations/flips + colour map, checked against the examples) | 6.4% | 0% |
| Local-rule lookup (3x3 neighbourhood -> colour) | 1.2% | 0% |
| Local rule, unseen neighbourhoods keep their colour | 3.5% | 0% |
| **Settle core, trained per task on its examples** (`settle_ttt.py`) | **5.8%** (7 of its 10 solved by no baseline) | **0.9%** (1 task) |
| Any baseline or Settle | 13.9% | 0.9% |

`python arc/arc_lab.py baseline --set dev|heldout`, `python arc/settle_ttt.py --set dev|heldout`.
Per-task results: `runs/arc/`.

## What we learned

- **Memorising is not learning the rule.** Settle reproduced all of its own examples on 91/173 dev and 52/114
  held-out tasks, but generalised on 10 and 1. Trained from scratch on 2-5 examples, a network has no reason
  to prefer the general rule over a lookup of the examples.
- **Augmentation must respect the rule.** Recolouring or rotating the examples stopped learning entirely: many
  ARC rules are about specific colours or directions, so the augmented examples contradicted each other.
- **Recurrence needs a gradient path.** Through 8-12 recurrent steps the gradient shrank about 50x and nothing
  was learned; a loss after every step (deep supervision) and an update initialised near "do nothing" fixed it.
- The held-out set has none of the simple rules that appear in the training set: every hand-built baseline
  scored 0.

## Next

A prior: train one Settle core across many tasks (slow learning), then adapt only a small per-task code from
the examples (fast learning), as a brain combines general skills with quick learning of a new situation.
Held-out tasks stay unseen until that method is frozen.
