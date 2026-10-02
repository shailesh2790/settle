"""ARC-style reasoning at laptop scale: learn a grid-transformation rule from 2-5 examples.

Data: ARC-AGI-1 (github.com/fchollet/ARC-AGI, Apache-2.0) in data/ARC-AGI.
Split (by rule, not by hand):
    dev       = ARC training tasks   whose outputs keep the input's size, grids <= 15x15   (173 tasks)
    heldout   = ARC evaluation tasks with the same rule                                    (114 tasks)
Design and tune on dev only; score held-out with frozen methods.
Scoring as in ARC Prize: a task counts only if EVERY test output is exactly right; up to 2 attempts.

    python arc/arc_lab.py split
    python arc/arc_lab.py baseline --set dev          # copy / transform search / local-rule lookup
"""
import argparse, glob, json, os, time
from itertools import product
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "ARC-AGI", "data")


# ---------------------------------------------------------------- data
def load_set(name, max_side=15):
    split = {"dev": "training", "heldout": "evaluation"}[name]
    tasks = []
    for f in sorted(glob.glob(os.path.join(DATA, split, "*.json"))):
        t = json.load(open(f)); pairs = t["train"] + t["test"]
        if not all(np.shape(p["input"]) == np.shape(p["output"]) for p in pairs): continue
        if max(max(np.shape(p["input"])) for p in pairs) > max_side: continue
        tasks.append({"id": os.path.basename(f)[:-5],
                      "train": [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]],
                      "test": [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]})
    return tasks


def score(task, attempts_per_test):
    """attempts_per_test: for each test input, a list of up to 2 candidate grids (or None)."""
    return all(any(c is not None and c.shape == y.shape and np.array_equal(c, y) for c in cands[:2])
               for (x, y), cands in zip(task["test"], attempts_per_test))


# ---------------------------------------------------------------- baselines
D8 = [lambda g: g, lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3),
      lambda g: g.T, lambda g: np.fliplr(g), lambda g: np.flipud(g), lambda g: np.rot90(g, 2).T]


def fit_colormap(pairs):
    """A consistent cell-wise colour map from transformed inputs to outputs, or None."""
    m = {}
    for x, y in pairs:
        if x.shape != y.shape: return None
        for a, b in zip(x.ravel(), y.ravel()):
            if m.setdefault(int(a), int(b)) != b: return None
    return m


def transform_search(task):
    """Geometric transform (8) optionally followed by a colour map; keep those that reproduce every example."""
    found = []
    for gi, g in enumerate(D8):
        tr = [(g(x), y) for x, y in task["train"]]
        if all(a.shape == y.shape and np.array_equal(a, y) for a, y in tr):
            found.append(lambda x, g=g: g(x))
        cm = fit_colormap(tr)
        if cm is not None:
            found.append(lambda x, g=g, cm=cm: np.vectorize(lambda v: cm.get(int(v), int(v)))(g(x)))
    return [[f(x) for f in found[:2]] or [None] for x, _ in task["test"]]


def local_rule(task, radius=1):
    """Learn 'neighbourhood -> output colour' from the examples (a hand-made cellular automaton)."""
    def patches(x):
        p = np.pad(x, radius, constant_values=-1)
        k = 2 * radius + 1
        return [tuple(p[i:i + k, j:j + k].ravel()) for i in range(x.shape[0]) for j in range(x.shape[1])]
    table = {}
    for x, y in task["train"]:
        for pt, v in zip(patches(x), y.ravel()):
            if table.setdefault(pt, int(v)) != v: return [[None] for _ in task["test"]]   # inconsistent rule
    out = []
    for x, _ in task["test"]:
        vals = [table.get(pt) for pt in patches(x)]
        out.append([None] if any(v is None for v in vals) else [np.array(vals).reshape(x.shape)])
    return out


def local_rule_fallback(task, radius=1):
    """Like local_rule, but unseen neighbourhoods keep their input colour and conflicts take the majority."""
    from collections import Counter, defaultdict
    def patches(x):
        p = np.pad(x, radius, constant_values=-1); k = 2 * radius + 1
        return [tuple(p[i:i + k, j:j + k].ravel()) for i in range(x.shape[0]) for j in range(x.shape[1])]
    votes = defaultdict(Counter)
    for x, y in task["train"]:
        for pt, v in zip(patches(x), y.ravel()): votes[pt][int(v)] += 1
    out = []
    for x, _ in task["test"]:
        vals = [votes[pt].most_common(1)[0][0] if pt in votes else int(c) for pt, c in zip(patches(x), x.ravel())]
        out.append([np.array(vals).reshape(x.shape)])
    return out


BASELINES = {"copy input": lambda t: [[x] for x, _ in t["test"]],
             "local rule + keep unseen": local_rule_fallback,
             "transform search": transform_search,
             "local-rule lookup": local_rule}


def run_baselines(name):
    tasks = load_set(name); print(f"{name}: {len(tasks)} tasks")
    solved_by = {}
    for bname, fn in BASELINES.items():
        t0 = time.time(); ok = [t["id"] for t in tasks if score(t, fn(t))]
        solved_by[bname] = set(ok)
        print(f"  {bname:<20} {len(ok):>3}/{len(tasks)}  ({len(ok) / len(tasks) * 100:4.1f}%)  {time.time() - t0:.1f}s")
    union = set().union(*solved_by.values())
    print(f"  {'any baseline':<20} {len(union):>3}/{len(tasks)}  ({len(union) / len(tasks) * 100:4.1f}%)")
    os.makedirs(os.path.join(ROOT, "runs", "arc"), exist_ok=True)
    json.dump({k: sorted(v) for k, v in solved_by.items()}, open(os.path.join(ROOT, "runs", "arc", f"baselines_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "baseline"])
    ap.add_argument("--set", default="dev", choices=["dev", "heldout"])
    a = ap.parse_args()
    if a.cmd == "split":
        for s in ("dev", "heldout"):
            ts = load_set(s); n_ex = [len(t["train"]) for t in ts]
            print(f"{s}: {len(ts)} tasks, {sum(len(t['test']) for t in ts)} test grids, {min(n_ex)}-{max(n_ex)} examples per task (median {int(np.median(n_ex))})")
    else:
        run_baselines(a.set)
