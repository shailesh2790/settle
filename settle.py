"""
Settle — a recurrent latent reasoner that thinks by iterating, not by emitting tokens.

    h0      = relu(conv(x))                       encode the problem once
    a_t     = relu(conv([h_t, x]))                recall: re-inject the problem every step
    h_{t+1} = h_t + conv(a_t)                     residual update, SAME weights every step
    answer  = conv(h_T)                           every cell answered in parallel

Four 3x3 kernels, ~12k parameters. Compute scales with how long it thinks, not with
how big it is. Pure NumPy with hand-derived gradients so it trains on any laptop CPU.
"""
import json, sys, time
import numpy as np

# ---------------------------------------------------------------- data
def make_maze(n, rng):
    S = 2 * n + 1
    g = np.ones((S, S), np.uint8)
    seen = np.zeros((n, n), bool); seen[0, 0] = True; g[1, 1] = 0
    stack = [(0, 0)]
    while stack:
        cx, cy = stack[-1]
        nb = [(cx + dx, cy + dy, dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= cx + dx < n and 0 <= cy + dy < n and not seen[cy + dy, cx + dx]]
        if not nb:
            stack.pop(); continue
        nx, ny, dx, dy = nb[rng.integers(len(nb))]
        seen[ny, nx] = True
        g[2 * cy + 1 + dy, 2 * cx + 1 + dx] = 0
        g[2 * ny + 1, 2 * nx + 1] = 0
        stack.append((nx, ny))
    for _ in range(50):
        a = rng.integers(n, size=2); b = rng.integers(n, size=2)
        if abs(a - b).sum() >= max(2, n / 2): break
    s = (2 * a[1] + 1, 2 * a[0] + 1); t = (2 * b[1] + 1, 2 * b[0] + 1)
    return g, s, t, shortest_path(g, s, t)

def shortest_path(g, s, t):
    S = g.shape[0]; prev = {s: None}; q = [s]; i = 0
    while i < len(q):
        u = q[i]; i += 1
        if u == t: break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < S and 0 <= v[1] < S and not g[v] and v not in prev:
                prev[v] = u; q.append(v)
    m = np.zeros_like(g, np.float32); u = t
    while u is not None: m[u] = 1; u = prev[u]
    return m

def batch(n, B, rng):
    S = 2 * n + 1
    X = np.zeros((B, S, S, 3), np.float32); Y = np.zeros((B, S, S, 1), np.float32)
    for b in range(B):
        g, s, t, p = make_maze(n, rng)
        X[b, :, :, 0] = g; X[b, s[0], s[1], 1] = 1; X[b, t[0], t[1], 2] = 1
        Y[b, :, :, 0] = p
    M = 1.0 - X[..., :1]
    return X, Y, M

# ---------------------------------------------------------------- conv via im2col
def im2col(x):
    B, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return np.concatenate([xp[:, dy:dy + H, dx:dx + W, :] for dy in range(3) for dx in range(3)], axis=-1).reshape(B * H * W, 9 * C)

def col2im(dcols, shape):
    B, H, W, C = shape
    d = dcols.reshape(B, H, W, 9, C); dxp = np.zeros((B, H + 2, W + 2, C), np.float32); k = 0
    for dy in range(3):
        for dx in range(3):
            dxp[:, dy:dy + H, dx:dx + W, :] += d[:, :, :, k, :]; k += 1
    return dxp[:, 1:H + 1, 1:W + 1, :]

def conv(cols, w, b, shape_out):
    return (cols @ w.reshape(-1, w.shape[-1]) + b).reshape(shape_out)

# ---------------------------------------------------------------- model
KEYS = ["wIn", "bIn", "w1", "b1", "w2", "b2", "wOut", "bOut"]

def init(C, seed=7):
    r = np.random.default_rng(seed)
    def glorot(shape, scale=1.0):
        fi = shape[0] * shape[1] * shape[2]; fo = shape[0] * shape[1] * shape[3]
        lim = np.sqrt(6 / (fi + fo)); return (r.uniform(-lim, lim, shape) * scale).astype(np.float32)
    return {"C": C, "wIn": glorot((3, 3, 3, C)), "bIn": np.zeros(C, np.float32),
            "w1": glorot((3, 3, C + 3, C)), "b1": np.zeros(C, np.float32),
            "w2": glorot((3, 3, C, C), 0.5), "b2": np.zeros(C, np.float32),
            "wOut": glorot((3, 3, C, 1)), "bOut": np.zeros(1, np.float32)}

def n_params(p): return int(sum(p[k].size for k in KEYS))

def encode(p, X):
    B, H, W, _ = X.shape; cx = im2col(X); z = conv(cx, p["wIn"], p["bIn"], (B, H, W, p["C"]))
    return np.maximum(z, 0), (cx, z)

def step(p, h, X, keep=False):
    B, H, W, C = h.shape
    c1 = im2col(np.concatenate([h, X], -1)); z1 = conv(c1, p["w1"], p["b1"], (B, H, W, C)); a = np.maximum(z1, 0)
    c2 = im2col(a); hn = np.tanh(h + conv(c2, p["w2"], p["b2"], (B, H, W, C)))
    return hn, ((c1, z1, c2, hn) if keep else None), a

def readout(p, h):
    B, H, W, C = h.shape; co = im2col(h); return conv(co, p["wOut"], p["bOut"], (B, H, W, 1)), co

def loss_and_grads(p, X, Y, M, T, n_pre):
    C = p["C"]; B, H, Wd, _ = X.shape; shp = (B, H, Wd, C)
    h, enc_cache = encode(p, X)
    for _ in range(n_pre): h, _, _ = step(p, h, X)
    caches = []
    for _ in range(T):
        h, c, _ = step(p, h, X, keep=True); caches.append(c)
    logit, co = readout(p, h)
    sig = 0.5 * (1 + np.tanh(0.5 * logit)); denom = M.sum() + 1e-6
    loss = float((M * (np.maximum(logit, 0) - logit * Y + np.log1p(np.exp(-np.abs(logit))))).sum() / denom)
    g = {k: np.zeros_like(p[k]) for k in KEYS}
    dl = ((sig - Y) * M / denom).reshape(-1, 1)
    g["wOut"] = (co.T @ dl).reshape(p["wOut"].shape); g["bOut"] = dl.sum(0)
    dh = col2im(dl @ p["wOut"].reshape(-1, 1).T, shp)
    W1 = p["w1"].reshape(-1, C); W2 = p["w2"].reshape(-1, C)
    for c1, z1, c2, hn in reversed(caches):
        dh = dh * (1 - hn * hn)                      # through tanh
        dhf = dh.reshape(-1, C)
        g["w2"] += (c2.T @ dhf).reshape(p["w2"].shape); g["b2"] += dhf.sum(0)
        da = col2im(dhf @ W2.T, shp); dz = (da * (z1 > 0)).reshape(-1, C)
        g["w1"] += (c1.T @ dz).reshape(p["w1"].shape); g["b1"] += dz.sum(0)
        dinp = col2im(dz @ W1.T, (B, H, Wd, C + 3))
        dh = dh + dinp[..., :C]
    if n_pre == 0:                       # encoder only receives gradient when the unroll starts from it
        cx, z0 = enc_cache; dz0 = (dh * (z0 > 0)).reshape(-1, C)
        g["wIn"] = (cx.T @ dz0).reshape(p["wIn"].shape); g["bIn"] = dz0.sum(0)
    return loss, g

# ---------------------------------------------------------------- inference with adaptive halting
def solve(p, X, max_iter=400, patience=4, trace=False):
    """Think until the answer stops changing for `patience` consecutive steps."""
    h, _ = encode(p, X); prev = None; stable = 0; history = []
    for t in range(1, max_iter + 1):
        h, _, a = step(p, h, X)
        pred = readout(p, h)[0] > 0
        if trace: history.append((pred.copy(), float((a > 0).mean())))
        if prev is not None and np.array_equal(pred, prev): stable += 1
        else: stable = 0
        prev = pred
        if stable >= patience: break
    return pred, t, history

def evaluate(p, n, count, rng, max_iter):
    solved = 0; steps = []; act = []
    for _ in range(count):
        X, Y, M = batch(n, 1, rng)
        pred, t, hist = solve(p, X, max_iter=max_iter, trace=True)
        ok = np.array_equal(pred[..., 0] & (M[..., 0] > 0), Y[..., 0] > 0)
        solved += ok; steps.append(t); act.append(np.mean([h[1] for h in hist]))
    return solved / count, float(np.mean(steps)), float(np.mean(act))

# ---------------------------------------------------------------- training
def adam_update(p, g, st, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    for k in KEYS:
        st["m"][k] = b1 * st["m"][k] + (1 - b1) * g[k]
        st["v"][k] = b2 * st["v"][k] + (1 - b2) * g[k] ** 2
        mh = st["m"][k] / (1 - b1 ** t); vh = st["v"][k] / (1 - b2 ** t)
        p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

def save(p, path, extra=None):
    out = {"C": p["C"]}
    for k in KEYS: out[k] = {"shape": list(p[k].shape), "data": [round(float(v), 5) for v in p[k].ravel()]}
    if extra: out["meta"] = extra
    json.dump(out, open(path, "w"))

def load(path):
    o = json.load(open(path)); p = {"C": o["C"]}
    for k in KEYS: p[k] = np.array(o[k]["data"], np.float32).reshape(o[k]["shape"])
    return p, o.get("meta", {})

def grad_check():
    rng = np.random.default_rng(0); p = init(4, seed=1); X, Y, M = batch(2, 2, rng)
    _, g = loss_and_grads(p, X, Y, M, T=3, n_pre=0); worst = 0
    for k in KEYS:
        flat = p[k].ravel()
        for i in rng.choice(flat.size, min(4, flat.size), replace=False):
            old = flat[i]; eps = 1e-3
            flat[i] = old + eps; lp, _ = loss_and_grads(p, X, Y, M, 3, 0)
            flat[i] = old - eps; lm, _ = loss_and_grads(p, X, Y, M, 3, 0)
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = g[k].ravel()[i]
            rel = abs(num - ana) / max(1e-4, abs(num) + abs(ana)); worst = max(worst, rel)
    return worst

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gradcheck":
        print("worst relative gradient error:", grad_check())
    elif cmd == "train":
        steps = int(sys.argv[2]); resume = len(sys.argv) > 3
        C, n, B, T, lr = 24, 5, 32, 24, 2e-3
        if resume:
            p, meta = load("settle_weights.json"); st = meta.pop("_opt", None); t0step = meta.get("step", 0)
        else:
            p, meta, t0step = init(C), {}, 0
        st = {"m": {k: np.zeros_like(p[k]) for k in KEYS}, "v": {k: np.zeros_like(p[k]) for k in KEYS}}
        if resume and __import__("os").path.exists("opt.npz"):
            z = np.load("opt.npz"); st = {"m": {k: z["m_" + k] for k in KEYS}, "v": {k: z["v_" + k] for k in KEYS}}
        rng = np.random.default_rng(100 + t0step); t = time.time(); log = meta.get("log", [])
        for i in range(1, steps + 1):
            n = int(rng.choice([4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.2]))
            T = int(4.5 * n + 4); Bn = 32 if n <= 5 else 20
            X, Y, M = batch(n, Bn, rng)
            n_pre = 0 if rng.random() < 0.3 else int(rng.integers(1, 2 * T))
            loss, g = loss_and_grads(p, X, Y, M, T, n_pre)
            gn = np.sqrt(sum(float((g[k] ** 2).sum()) for k in KEYS))
            if gn > 1.0:
                for k in KEYS: g[k] *= 1.0 / gn
            step_no = t0step + i
            adam_update(p, g, st, lr * (0.5 if step_no > 1500 else 1.0), step_no)
            if step_no % 25 == 0:
                log.append([step_no, round(loss, 4)])
                print(f"step {step_no} loss {loss:.4f} gnorm {gn:.2f} {(time.time() - t) / i:.2f}s/step", flush=True)
        meta.update({"step": t0step + steps, "log": log, "trainN": "4-7 (9x9 to 15x15)", "T": "4.5n+4", "C": C, "state": "tanh"})
        save(p, "settle_weights.json", meta)
        np.savez("opt.npz", **{"m_" + k: st["m"][k] for k in KEYS}, **{"v_" + k: st["v"][k] for k in KEYS})
        print("saved step", t0step + steps)
    elif cmd == "eval":
        p, meta = load("settle_weights.json"); rng = np.random.default_rng(12345)
        plan = {"5": (5, 100, 80), "8": (8, 60, 200), "10": (10, 50, 300), "15": (15, 30, 500), "20": (20, 20, 700)}
        for key in sys.argv[2:]:
            n, cnt, mx = plan[key]
            acc, st_, act = evaluate(p, n, cnt, np.random.default_rng(1000 + n), mx)
            print(f"maze {2*n+1}x{2*n+1}: solved {acc*100:.0f}%  mean thinking steps {st_:.1f}  active units {act*100:.0f}%", flush=True)
