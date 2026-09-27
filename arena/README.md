# The Arena — one scorecard for every idea

Every entrant solves the same procedurally generated instances. It trains on easy ones and is tested on
harder ones. The scorecard reports, per difficulty: exact-solve rate, inference MFLOPs per instance,
solves per GFLOP, and **calibration AUROC** — does the entrant's own confidence tell right answers from wrong ones?

## Results so far (mazes; everything trained on 9x9 to 15x15, one CPU core)

Rerun 2026-09-27 on identical instances for every entrant, after fixing the AUROC tie bug (see below).

| Maze | Entrant | Solved | MFLOP/instance | Solves/GFLOP | AUROC |
|---|---|---|---|---|---|
| 11x11 | settle (recurrent latent) | 95% | 46 | 20.5 | 0.93 |
| 11x11 | energy (descent on answer) | 8% | 65 | 1.2 | 1.00 |
| 11x11 | settle + energy verifier | 95% | 47 | 20.2 | 1.00 |
| 11x11 | settle + longer only when unsure | 95% | 46 | 20.5 | 0.93 |
| 11x11 | settle + think more when unsure | 98% | 85 | 11.4 | 0.50 |
| 17x17 | settle | 100% | 214 | 4.67 | n/a |
| 17x17 | settle + longer only when unsure | 100% | 214 | 4.67 | n/a |
| 17x17 | settle + think more when unsure | 97% | 1,474 | 0.66 | 0.48 |
| 21x21 | settle | 90% | 769 | 1.17 | 0.99 |
| 21x21 | energy | 0% | 554 | 0 | n/a |
| 21x21 | settle + energy verifier | 90% | 771 | 1.17 | 0.88 |
| 21x21 | settle + longer only when unsure | 90% | 1,033 | 0.87 | 0.99 |
| 21x21 | settle + think more when unsure | 87% | 6,449 | 0.13 | 0.60 |
| 31x31 | settle | 75% | 4,988 | 0.15 | 1.00 |
| 31x31 | settle + energy verifier | 75% | 4,994 | 0.15 | 0.81 |
| 31x31 | settle + longer only when unsure | 83% | 8,638 | 0.10 | 0.95 |
| 31x31 | settle + think more when unsure | 83% | 74,575 | 0.01 | 0.80 |
| 41x41 | settle | 25% | 34,989 | 0.007 | 0.93 |
| 41x41 | settle + longer only when unsure | 33% | 123,328 | 0.003 | 0.88 |
| 41x41 | settle + think more when unsure | 50% | 765,187 | 0.0008 | 0.88 |

Instances: 40 / 30 / 30 / 12 / 12 per size (small at 31x31 and 41x41: one instance = 8 points).
Full table: `python arena.py report`.

**AUROC fix.** The original scorer ranked confidences with `argsort(argsort(...))`, which breaks
ties arbitrarily. Settle's confidence is an integer, so ties are common: an uninformative confidence
scored 0.33 or 0.67 depending on instance order instead of 0.5. It now uses
P(conf right > conf wrong) + 0.5 P(tie). This moved 31x31 from 0.96/0.96 (settle / energy
verifier) to 1.00/0.81.

**What the entrants show.**
- Energy descent does not solve mazes beyond 11x11, and its learned verifier degrades with size
  (1.00, 0.88, 0.81). Settle's own confidence stays at 0.93-1.00, even beyond its training sizes.
- *Longer only when unsure* is a clean win: identical cost when Settle is confident, +8 points at
  31x31 and 41x41, and calibration is kept.
- *Think more when unsure* (wall off an uncertain cell in a copy of the maze, re-think, keep the
  most confident copy) raises 41x41 from 25% to 50% but breaks correct answers at 17x17 and 21x21,
  and calibration collapses to 0.48-0.80. Settle's confidence is trustworthy for its own answer on
  a real maze, not for answers on modified mazes it never saw (some copies have no path at all).
  A search needs a judge that stays valid on the problems it creates. The Sudoku search works because
  its judge is an exact rule check.
- **Fix: teach Settle to say "no path"** (`settle_nopath.py`). Fine-tuning for 1,500 steps on a mix
  with 25% unsolvable mazes (one path cell walled off) makes Settle recognise 80/80 unsolvable mazes
  up to 31x31 (the original weights: 0/80; they always draw a path). The search judge then requires the
  model to be confident AND to claim a path (mark start and goal), both from its own output:

  | Maze | Settle | old search | knows "no path" | knows "no path" + search | MFLOP (new search) |
  |---|---|---|---|---|---|
  | 11x11 | 95% | 97.5% | 100% | 100% | 102 |
  | 17x17 | 100% | 96.7% | 93.3% | 93.3% | 534 |
  | 21x21 | 90% | 86.7% | 93.3% | 96.7% | 1,554 |
  | 31x31 | 75% | 83.3% | 75.0% | 83.3% | 15,596 |
  | 41x41 | 25% | 50.0% | 50.0% | 58.3% | 113,886 (old search: 765,187) |

  The search no longer does worse than the model it starts from at any size, its calibration is
  AUROC 1.00 wherever defined, and it is 3-7x cheaper than the old search. Cost of the fine-tune: two
  17x17 mazes (100% -> 93.3%), which it now answers with "no path" rather than a wrong path. Sample sizes
  at 31x31 and 41x41 are 12 mazes (one maze = 8 points).
- `diagnose.py` breaks failures down: at 41x41, 7 of 9 failures hit the step cap; the others settle
  on a wrong branch or a half-filled path.

## Plugging in a new idea

```python
class MyEntrant:
    name = "my idea"
    params = 12345            # parameter count
    train_flops = 1.2e15      # approximate training compute
    def solve(self, X):       # X: (1, H, W, 3) walls / start / goal
        ...
        return pred_bool_HxW, {"flops": ..., "steps": ..., "conf": ...}
```
Add it to the `ents` list in `arena.py`. For a new task family, add a `make(n, rng) -> (X, Y)` to `TASKS`.

## Files

- `settle.py`, `settle.json` — recurrent latent reasoner (option 1), trained weights
- `ebm.py`, `ebm.json` — energy-based reasoner (option 2), trained weights; `python ebm.py 1000` continues training
- `gradcheck_ebm.py` — float64 check of the energy model's hand-written gradients
- `arena.py`, `arena_results.json` — harness and results (`ARENA_ONLY=<name part>` runs a subset)
- `diagnose.py` — per-instance failure breakdown for Settle on large mazes
- `settle_nopath.py`, `settle_nopath.json` — fine-tune that teaches Settle to say "no path"; `python settle_nopath.py check`
