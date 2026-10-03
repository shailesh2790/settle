"""ARC with the Local Field Learner: learn each task's rule with ONE closed-form ridge solve.

Every cell is described by the colours in its (2r+1)x(2r+1) neighbourhood (one-hot, with "outside the grid"
as an 11th colour), passed through the fixed random HD encoder (lfl/local_field.py), and a ridge readout maps
that code to the output colour. Fitting a task is a single linear solve on its example cells, so there is no
gradient descent and no training loop. Unlike an exact lookup, similar neighbourhoods give similar answers.

    python arc/hd_ridge.py tune                         # grid search on dev only
    python arc/hd_ridge.py run --set heldout --r 1 --D 2048 --lam 10 --r2 2   # frozen config, run once
"""
import argparse, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arc_lab import load_set, score, ROOT
from lfl.local_field import HDEncoder


def neighbourhoods(x, r):
    """One-hot colours of each cell's neighbourhood: (cells, (2r+1)^2 * 11)."""
    p = np.pad(x, r, constant_values=10); k = 2 * r + 1; H, W = x.shape
    win = np.stack([p[i:i + H, j:j + W] for i in range(k) for j in range(k)], -1).reshape(H * W, k * k)
    return np.eye(11, dtype=np.float32)[win].reshape(H * W, -1)


def ridge_predict(F, Y, Ft, lam):
    """Ridge regression, solved in whichever form is smaller (same answer either way)."""
    n, d = F.shape
    if n < d:                                                   # dual form: invert an n x n matrix
        K = F @ F.T; alpha = np.linalg.solve(K + lam * np.eye(n), Y)
        return Ft @ (F.T @ alpha)
    return Ft @ np.linalg.solve(F.T @ F + lam * np.eye(d), F.T @ Y)


def solve_task(task, r, D, lam, seed=0):
    X = np.concatenate([neighbourhoods(x, r) for x, _ in task["train"]])
    y = np.concatenate([yy.ravel() for _, yy in task["train"]])
    enc = HDEncoder(X.shape[1], D=D, seed=seed).fit(X)
    F = enc(X).astype(np.float64); Y = -np.ones((len(y), 10)); Y[np.arange(len(y)), y] = 1
    preds = []
    for x, _ in task["test"]:
        S = ridge_predict(F, Y, enc(neighbourhoods(x, r)).astype(np.float64), lam)
        preds.append(S.argmax(1).reshape(x.shape))
    return preds


def evaluate(tasks, cfgs):
    """cfgs: list of (r, D, lam); each config gives one attempt. Returns solved task ids."""
    solved = []
    for t in tasks:
        per_cfg = [solve_task(t, *c) for c in cfgs]
        attempts = [[p[i] for p in per_cfg] for i in range(len(t["test"]))]
        if score(t, attempts): solved.append(t["id"])
    return solved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "run"]); ap.add_argument("--set", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--r", type=int, default=1); ap.add_argument("--D", type=int, default=2048); ap.add_argument("--lam", type=float, default=10)
    ap.add_argument("--r2", type=int, help="neighbourhood radius for the second attempt (same D and lam)")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "runs", "arc"), exist_ok=True)
    if a.cmd == "tune":
        dev = load_set("dev"); rows = []
        for r, D, lam in itertools.product((1, 2), (1024, 4096), (1, 10, 100)):
            t0 = time.time(); s = evaluate(dev, [(r, D, lam)])
            rows.append({"r": r, "D": D, "lam": lam, "solved": len(s), "ids": s, "secs": round(time.time() - t0, 1)})
            print(f"r={r} D={D:<5} lam={lam:<4} dev solved {len(s):>3}/{len(dev)}  {time.time() - t0:5.1f}s", flush=True)
        json.dump(rows, open(os.path.join(ROOT, "runs", "arc", "hd_ridge_tune_dev.json"), "w"), indent=1)
    else:
        tasks = load_set(a.set); cfgs = [(a.r, a.D, a.lam)] + ([(a.r2, a.D, a.lam)] if a.r2 else [])
        t0 = time.time(); s = evaluate(tasks, cfgs)
        print(f"{a.set}: solved {len(s)}/{len(tasks)} ({len(s) / len(tasks) * 100:.1f}%) with attempts {cfgs} in {time.time() - t0:.0f}s: {s}")
        json.dump({"set": a.set, "configs": cfgs, "solved": s, "n": len(tasks)},
                  open(os.path.join(ROOT, "runs", "arc", f"hd_ridge_{a.set}.json"), "w"), indent=1)
