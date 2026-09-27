"""Teach Settle to say "there is no path", so its judgement stays valid on the altered mazes a search creates.

Fine-tunes the shipped maze weights (settle.json) on a mix of normal mazes and UNSOLVABLE ones: a perfect
maze is a tree, so walling off any cell on the start-goal path disconnects them, and the right answer
becomes "no path cells at all". Settle then claims a path (marks start and goal) only when one exists.

    python settle_nopath.py train 1500      # fine-tune -> settle_nopath.json (about 15 min on one CPU core)
    python settle_nopath.py check           # solvable accuracy and "no path" detection, old vs new weights
"""
import json, sys, time
import numpy as np
import settle as S


def wall_off_path(g, s, t, path, rng):
    """Wall one path cell (never an endpoint): the maze becomes unsolvable."""
    cells = [tuple(c) for c in np.argwhere(path > 0) if tuple(c) != s and tuple(c) != t]
    c = cells[rng.integers(len(cells))]
    g = g.copy(); g[c] = 1
    return g


def batch_mixed(n, B, rng, frac_unsolvable):
    Sz = 2 * n + 1
    X = np.zeros((B, Sz, Sz, 3), np.float32); Y = np.zeros((B, Sz, Sz, 1), np.float32)
    for b in range(B):
        g, s, t, path = S.make_maze(n, rng)
        if rng.random() < frac_unsolvable:
            g = wall_off_path(g, s, t, path, rng); path = np.zeros_like(path)
        X[b, :, :, 0] = g; X[b, s[0], s[1], 1] = 1; X[b, t[0], t[1], 2] = 1; Y[b, :, :, 0] = path
    return X, Y, 1.0 - X[..., :1]


def claims_path(pred, X):
    """The model's own verdict: does it mark both endpoints as part of a path?"""
    s = np.argwhere(X[0, ..., 1] > 0)[0]; t = np.argwhere(X[0, ..., 2] > 0)[0]
    return bool(pred[tuple(s)] and pred[tuple(t)])


def think(p, X, cap, patience=4):
    h, _ = S.encode(p, X); prev = None; stable = 0
    for step in range(1, cap + 1):
        h, _, _ = S.step(p, h, X)
        logit = S.readout(p, h)[0][0, ..., 0]
        pred = (logit > 0) & (X[0, ..., 0] == 0)
        stable = stable + 1 if prev is not None and np.array_equal(pred, prev) else 0
        prev = pred
        if stable >= patience: break
    return pred


def train(steps, src="settle.json", out="settle_nopath.json", frac=0.25, lr=1e-3, seed=7):
    p, meta = S.load(src)
    st = {"m": {k: np.zeros_like(p[k]) for k in S.KEYS}, "v": {k: np.zeros_like(p[k]) for k in S.KEYS}}
    rng = np.random.default_rng(seed); t0 = time.time(); log = []
    for i in range(1, steps + 1):
        n = int(rng.choice([4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.2]))           # same curriculum as settle.py
        T = int(4.5 * n + 4); B = 32 if n <= 5 else 20
        X, Y, M = batch_mixed(n, B, rng, frac)
        n_pre = 0 if rng.random() < 0.3 else int(rng.integers(1, 2 * T))
        loss, g = S.loss_and_grads(p, X, Y, M, T, n_pre)
        gn = np.sqrt(sum(float((g[k] ** 2).sum()) for k in S.KEYS))
        if gn > 1.0:
            for k in S.KEYS: g[k] *= 1.0 / gn
        S.adam_update(p, g, st, lr * (0.5 if i > steps * 0.6 else 1.0), i)
        if i % 50 == 0:
            log.append([i, round(loss, 4)])
            print(f"step {i} loss {loss:.4f} {(time.time() - t0) / i:.2f}s/step", flush=True)
    S.save(p, out, {"base": src, "finetune_steps": steps, "frac_unsolvable": frac, "log": log})
    print("saved", out)


def check(weights=("settle.json", "settle_nopath.json"), sizes=(5, 8, 10, 15), count=20):
    for w in weights:
        p, _ = S.load(w)
        print(f"\n{w}")
        for n in sizes:
            H = 2 * n + 1; rng = np.random.default_rng(5000 + n); solved = claimed_ok = said_none = 0
            for _ in range(count):
                g, s, t, path = S.make_maze(n, rng)
                X = np.zeros((1, H, H, 3), np.float32); X[0, :, :, 0] = g; X[0, s[0], s[1], 1] = 1; X[0, t[0], t[1], 2] = 1
                pred = think(p, X, 30 * H); solved += np.array_equal(pred, path > 0); claimed_ok += claims_path(pred, X)
                Xu = X.copy(); Xu[0, :, :, 0] = wall_off_path(g, s, t, path, rng)
                said_none += not claims_path(think(p, Xu, 30 * H), Xu)
            print(f"  {H}x{H}: solvable solved {solved}/{count}, claims a path on solvable {claimed_ok}/{count}, "
                  f"says 'no path' on unsolvable {said_none}/{count}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train": train(int(sys.argv[2]))
    else: check()
