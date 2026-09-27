"""Energy-based reasoner (option 2), NumPy, hand-derived gradients.

  E(x, y) = sum over cells of  w3 . relu(conv(relu(conv([x, y]))))      a small, LOCAL energy
  think   : y <- project( y - eta * dE/dy / rms(dE/dy) )                 gradient descent on the ANSWER
  confidence = -E(x, round(y)) per open cell                             the energy doubles as a self-check

Why a local energy can work for mazes: in a perfect maze (a tree) the solution is the only set of open
cells where start and goal have one path-neighbour and every other chosen cell has exactly two.
Those are local constraints a 5x5 receptive field can check. Whether gradient descent finds that
minimum from a blank start is exactly the question the arena answers.

Training (no second-order gradients): contrastive logistic loss pushing E(x, y_true) below
  (a) corrupted answers (random flips) and (b) the model's own partially-descended answers."""
import json, sys, time
import numpy as np
from settle import im2col, col2im, make_maze

KEYS = ["w1", "b1", "w2", "b2", "w3", "b3"]


def init(C=16, seed=3):
    r = np.random.default_rng(seed)
    g = lambda s: (r.standard_normal(s) * np.sqrt(2.0 / (s[0] * s[1] * s[2]))).astype(np.float32)
    return {"C": C, "w1": g((3, 3, 4, C)), "b1": np.zeros(C, np.float32), "w2": g((3, 3, C, C)),
            "b2": np.zeros(C, np.float32), "w3": (r.standard_normal((C, 1)) * 0.1).astype(np.float32),
            "b3": np.zeros(1, np.float32)}


def n_params(p): return int(sum(p[k].size for k in KEYS))


def forward(p, X, y):
    B, H, W, _ = X.shape; C = p["C"]
    c1 = im2col(np.concatenate([X, y], -1)); z1 = (c1 @ p["w1"].reshape(-1, C) + p["b1"]).reshape(B, H, W, C); a1 = np.maximum(z1, 0)
    c2 = im2col(a1); z2 = (c2 @ p["w2"].reshape(-1, C) + p["b2"]).reshape(B, H, W, C); a2 = np.maximum(z2, 0)
    e = (a2.reshape(-1, C) @ p["w3"] + p["b3"]).reshape(B, H, W)
    open_ = 1.0 - X[..., 0]
    E = (e * open_).sum((1, 2))                       # only open cells contribute
    return E, (c1, z1, c2, z2, a2, open_)


def backward(p, X, cache, gE, want_params=True):
    """gE: (B,) upstream dL/dE. Returns (param grads or None, dL/dy)."""
    c1, z1, c2, z2, a2, open_ = cache
    B, H, W, C = z1.shape
    de = (gE[:, None, None] * open_).reshape(-1, 1).astype(np.float32)
    g = {}
    if want_params:
        g["w3"] = a2.reshape(-1, C).T @ de; g["b3"] = de.sum(0)
    dz2 = (de @ p["w3"].T).reshape(B, H, W, C) * (z2 > 0)
    dz2f = dz2.reshape(-1, C)
    if want_params:
        g["w2"] = (c2.T @ dz2f).reshape(p["w2"].shape); g["b2"] = dz2f.sum(0)
    da1 = col2im(dz2f @ p["w2"].reshape(-1, C).T, (B, H, W, C))
    dz1f = (da1 * (z1 > 0)).reshape(-1, C)
    if want_params:
        g["w1"] = (c1.T @ dz1f).reshape(p["w1"].shape); g["b1"] = dz1f.sum(0)
    dinp = col2im(dz1f @ p["w1"].reshape(-1, C).T, (B, H, W, 4))
    return (g if want_params else None), dinp[..., 3:4]


def project(y, X):
    walls, s, t = X[..., :1], X[..., 1:2], X[..., 2:3]
    y = np.clip(y, 0, 1) * (1 - walls)
    return np.maximum(y, np.maximum(s, t))           # endpoints are always on the path


def init_answer(X): return project(np.full(X[..., :1].shape, 0.5, np.float32), X)


def think(p, X, steps, eta=0.15, noise=0.0, rng=None, patience=4, trace=False, binarize=1.0):
    """Gradient descent on the answer with an annealed step and a growing pull toward 0/1.
    Halts when the rounded answer is stable for `patience` steps (after a short warm-up)."""
    y = init_answer(X); prev = None; stable = 0; hist = []
    for t in range(1, steps + 1):
        frac = t / steps
        E, cache = forward(p, X, y)
        _, gy = backward(p, X, cache, np.ones(X.shape[0], np.float32), want_params=False)
        rms = np.sqrt((gy ** 2).mean((1, 2, 3), keepdims=True)) + 1e-8
        step = eta * (1 - 0.9 * frac)
        y = y - step * (gy / rms + binarize * frac * (1 - 2 * y))
        if noise and rng is not None: y = y + noise * (1 - frac) * rng.standard_normal(y.shape).astype(np.float32)
        y = project(y, X)
        pred = y > 0.5
        if trace: hist.append(pred.copy())
        if prev is not None and np.array_equal(pred, prev): stable += 1
        else: stable = 0
        prev = pred
        if stable >= patience and frac > 0.5: break
    return y, t, hist


def energy_confidence(p, X, y):
    yb = (y > 0.5).astype(np.float32)
    E, _ = forward(p, X, yb)
    return -E / (1.0 - X[..., 0]).sum((1, 2))


# ------------------------------------------------------------------ training
def batch(sizes, B, rng):
    n = int(rng.choice(sizes)); S = 2 * n + 1
    X = np.zeros((B, S, S, 3), np.float32); Y = np.zeros((B, S, S, 1), np.float32)
    for b in range(B):
        g, s, t, path = make_maze(n, rng)
        X[b, :, :, 0] = g; X[b, s[0], s[1], 1] = 1; X[b, t[0], t[1], 2] = 1; Y[b, :, :, 0] = path
    return X, Y


def corrupt(X, Y, rng):
    y = Y.copy(); B = y.shape[0]; openm = (X[..., :1] == 0)
    for b in range(B):
        idx = np.flatnonzero(openm[b].ravel()); k = int(rng.integers(1, 7))
        pick = rng.choice(idx, min(k, len(idx)), replace=False)
        flat = y[b].ravel(); flat[pick] = 1 - flat[pick]
    return project(y, X)


def train(p, steps, log_every=25, seed=0, lr=2e-3, state=None):
    rng = np.random.default_rng(seed)
    st = state or {"m": {k: np.zeros_like(p[k]) for k in KEYS}, "v": {k: np.zeros_like(p[k]) for k in KEYS}, "t": 0}
    t0 = time.time()
    for i in range(steps):
        X, Y = batch([4, 5, 6], 32, rng)
        # negatives: half corrupted truths, half the model's own partial descent from a blank answer
        yneg = corrupt(X, Y, rng)
        k = int(rng.integers(10, 70))
        ys, _, _ = think(p, X[16:], k, noise=0.05, rng=rng, patience=10**6)
        yneg[16:] = ys
        same = np.all((yneg > 0.5) == (Y > 0.5), axis=(1, 2, 3))          # a "negative" that is actually right
        Ep, cp = forward(p, X, Y); En, cn = forward(p, X, yneg)
        d = np.clip(Ep - En, -30, 30); sig = 1 / (1 + np.exp(-d))
        w = (~same).astype(np.float32)
        loss = float((w * np.log1p(np.exp(d))).sum() / max(w.sum(), 1) + 1e-3 * ((Ep ** 2).mean() + (En ** 2).mean()))
        gP = (w * sig / max(w.sum(), 1) + 2e-3 * Ep / len(Ep)).astype(np.float32)
        gN = (-w * sig / max(w.sum(), 1) + 2e-3 * En / len(En)).astype(np.float32)
        g1, _ = backward(p, X, cp, gP); g2, _ = backward(p, X, cn, gN)
        g = {k: g1[k] + g2[k] for k in KEYS}
        gn = np.sqrt(sum(float((g[k] ** 2).sum()) for k in KEYS))
        if gn > 1: g = {k: g[k] / gn for k in KEYS}
        st["t"] += 1
        for kk in KEYS:
            st["m"][kk] = 0.9 * st["m"][kk] + 0.1 * g[kk]; st["v"][kk] = 0.999 * st["v"][kk] + 0.001 * g[kk] ** 2
            p[kk] -= (lr * (st["m"][kk] / (1 - 0.9 ** st["t"])) / (np.sqrt(st["v"][kk] / (1 - 0.999 ** st["t"])) + 1e-8)).astype(np.float32)
        if (i + 1) % log_every == 0:
            acc = float((Ep < En)[w > 0].mean()) if w.sum() else 1.0
            print(f"step {st['t']} loss {loss:.4f} margin-ok {acc:.2f} {(time.time()-t0)/(i+1):.2f}s/step", flush=True)
    return st


def save(p, path, st=None, extra=None):
    o = {"C": p["C"], **{k: {"shape": list(p[k].shape), "data": [round(float(v), 6) for v in p[k].ravel()]} for k in KEYS}}
    o["meta"] = extra or {}
    json.dump(o, open(path, "w"))
    if st: np.savez(path + ".opt.npz", t=st["t"], **{"m_" + k: st["m"][k] for k in KEYS}, **{"v_" + k: st["v"][k] for k in KEYS})


def load(path, with_opt=False):
    o = json.load(open(path)); p = {"C": o["C"]}
    for k in KEYS: p[k] = np.array(o[k]["data"], np.float32).reshape(o[k]["shape"])
    if not with_opt: return p, o.get("meta", {})
    z = np.load(path + ".opt.npz")
    st = {"t": int(z["t"]), "m": {k: z["m_" + k] for k in KEYS}, "v": {k: z["v_" + k] for k in KEYS}}
    return p, o.get("meta", {}), st


if __name__ == "__main__":
    import os
    steps = int(sys.argv[1]); path = "ebm.json"
    if os.path.exists(path):
        p, meta, st = load(path, with_opt=True)
    else:
        p, meta, st = init(), {"steps": 0}, None
    st = train(p, steps, seed=meta.get("steps", 0) + 1, state=st)
    meta["steps"] = meta.get("steps", 0) + steps
    save(p, path, st, meta); print("saved", meta["steps"])
