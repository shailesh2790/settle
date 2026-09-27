"""
Settle in PyTorch — same model as settle.py, but GPU-ready and widenable.

    python settle_torch.py parity                      # check against the NumPy model on shipped weights
    python settle_torch.py train --C 64 --steps 20000  # wider model, trained on 9x9..21x21
    python settle_torch.py eval  --weights settle_torch_weights.json 5 8 10 15 20
    python build_web.py settle_torch_weights.json      # put the new model in the browser page

Weights are saved in settle.py's JSON format (HWIO kernels), so all three front ends share them.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settle   # maze generation, BFS ground truth and the reference NumPy model

KEYS = settle.KEYS


class Settle(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.inp = nn.Conv2d(3, C, 3, padding=1)
        self.c1 = nn.Conv2d(C + 3, C, 3, padding=1)
        self.c2 = nn.Conv2d(C, C, 3, padding=1)
        self.out = nn.Conv2d(C, 1, 3, padding=1)
        with torch.no_grad(): self.c2.weight.mul_(0.5)   # same small-residual start as settle.init

    def encode(self, x): return F.relu(self.inp(x))

    def step(self, h, x):
        a = F.relu(self.c1(torch.cat([h, x], 1)))
        return torch.tanh(h + self.c2(a)), a

    def readout(self, h): return self.out(h)

    # ---- interchange with settle.py's JSON (HWIO kernels) ----
    _map = {"wIn": "inp.weight", "bIn": "inp.bias", "w1": "c1.weight", "b1": "c1.bias",
            "w2": "c2.weight", "b2": "c2.bias", "wOut": "out.weight", "bOut": "out.bias"}

    @classmethod
    def from_json(cls, path):
        o = json.load(open(path)); m = cls(o["C"]); sd = {}
        for k, name in cls._map.items():
            a = torch.tensor(np.array(o[k]["data"], np.float32).reshape(o[k]["shape"]))
            sd[name] = a.permute(3, 2, 0, 1).contiguous() if a.dim() == 4 else a
        m.load_state_dict(sd); return m, o.get("meta", {})

    def to_json(self, path, meta=None):
        sd = self.state_dict(); out = {"C": self.C}
        for k, name in self._map.items():
            a = sd[name].detach().float().cpu()
            if a.dim() == 4: a = a.permute(2, 3, 1, 0)                   # OIHW -> HWIO
            out[k] = {"shape": list(a.shape), "data": [round(float(v), 5) for v in a.numpy().ravel()]}
        if meta: out["meta"] = meta
        json.dump(out, open(path, "w"))


def to_nchw(X, Y, M, dev):
    f = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(dev)
    return f(X), f(Y), f(M)


@torch.no_grad()
def solve(model, x, max_iter, patience=4):
    """Batched adaptive halting: each maze stops when its answer is unchanged for `patience` steps."""
    B = x.shape[0]; h = model.encode(x)
    prev = None; stable = torch.zeros(B, dtype=torch.long, device=x.device)
    final = torch.zeros_like(x[:, :1], dtype=torch.bool); steps = torch.full((B,), max_iter, device=x.device)
    done = torch.zeros(B, dtype=torch.bool, device=x.device)
    for t in range(1, max_iter + 1):
        h, _ = model.step(h, x)
        pred = model.readout(h) > 0
        if prev is not None:
            same = (pred == prev).flatten(1).all(1)
            stable = torch.where(same, stable + 1, torch.zeros_like(stable))
        prev = pred
        newly = (stable >= patience) & ~done
        final[newly] = pred[newly]; steps[newly] = t; done |= newly
        if done.all(): break
    final[~done] = pred[~done]
    return final, steps


def evaluate(model, n, count, max_iter, dev, seed):
    rng = np.random.default_rng(seed); solved = 0; steps = []
    for i in range(0, count, 16):
        X, Y, M = settle.batch(n, min(16, count - i), rng)
        x, y, m = to_nchw(X, Y, M, dev)
        pred, st = solve(model, x, max_iter)
        ok = ((pred & (m > 0)) == (y > 0)).flatten(1).all(1)
        solved += int(ok.sum()); steps += st.tolist()
    return solved / count, float(np.mean(steps))


EVAL_PLAN = {5: (100, 120), 8: (60, 250), 10: (50, 400), 15: (30, 800), 20: (20, 1600), 25: (12, 2500)}


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.resume:
        model, meta = Settle.from_json(a.out); start = meta.get("step", 0)
    else:
        model, meta, start = Settle(a.C), {}, 0
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sizes = list(range(a.nmin, a.nmax + 1))
    print(f"device {dev}  C={model.C}  params {sum(p.numel() for p in model.parameters()):,}  mazes {2*a.nmin+1}..{2*a.nmax+1}", flush=True)
    t0 = time.time(); log = meta.get("log", [])
    for i in range(1, a.steps + 1):
        step_no = start + i
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, i / a.steps)))      # cosine decay
        for g in opt.param_groups: g["lr"] = lr
        n = int(rng.choice(sizes)); T = int(4.5 * n + 4)
        B = max(8, int(a.batch * (a.nmin / n) ** 2))                          # constant-ish cells per batch
        x, y, m = to_nchw(*settle.batch(n, B, rng), dev)
        # Incremental progress training: start from a detached state reached after a random number
        # of steps, so the update learns to improve ANY state rather than memorise one trajectory.
        n_pre = 0 if rng.random() < 0.3 else int(rng.integers(1, 2 * T))
        h = model.encode(x)
        if n_pre:
            with torch.no_grad():
                for _ in range(n_pre): h, _ = model.step(h, x)
        for _ in range(T): h, _ = model.step(h, x)
        logit = model.readout(h)
        loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / (m.sum() + 1e-6)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step()
        if step_no % a.log_every == 0:
            log.append([step_no, round(float(loss), 4)])
            print(f"step {step_no} n={n} loss {float(loss):.4f} gnorm {gn:.2f} lr {lr:.2e} {(time.time() - t0) / i:.2f}s/step", flush=True)
        if step_no % a.save_every == 0 or i == a.steps:
            meta.update({"step": step_no, "log": log, "trainN": f"{a.nmin}-{a.nmax}", "T": "4.5n+4", "C": model.C, "state": "tanh"})
            model.to_json(a.out, meta)
    print("saved", a.out, "step", start + a.steps)


def parity():
    """The PyTorch model must reproduce settle.py exactly on the shipped weights."""
    root = os.path.dirname(os.path.abspath(__file__))
    p, _ = settle.load(os.path.join(root, "settle_weights.json"))
    model, _ = Settle.from_json(os.path.join(root, "settle_weights.json"))
    X, Y, M = settle.batch(8, 3, np.random.default_rng(3))
    h, _ = settle.encode(p, X)
    for _ in range(20): h, _, _ = settle.step(p, h, X)
    ref = settle.readout(p, h)[0]
    x, _, _ = to_nchw(X, Y, M, "cpu")
    with torch.no_grad():
        ht = model.encode(x)
        for _ in range(20): ht, _ = model.step(ht, x)
        got = model.readout(ht).permute(0, 2, 3, 1).numpy()
    err = float(np.abs(got - ref).max())
    # round trip through JSON must be lossless (to 5 decimals)
    tmp = os.path.join(root, "_roundtrip.json"); model.to_json(tmp); m2, _ = Settle.from_json(tmp); os.remove(tmp)
    rt = max(float((a - b).abs().max()) for a, b in zip(model.state_dict().values(), m2.state_dict().values()))
    print(f"max |logit torch - numpy| after 20 steps: {err:.2e}   JSON round-trip error: {rt:.1e}")
    return err < 1e-3 and rt < 1e-4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parity", "train", "eval"])
    ap.add_argument("sizes", nargs="*", type=int, help="eval: maze half-sizes n (maze is 2n+1)")
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default="settle_torch_weights.json")
    ap.add_argument("--weights", default="settle_weights.json")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "parity":
        sys.exit(0 if parity() else 1)
    elif a.cmd == "train":
        train(a)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = Settle.from_json(a.weights); model.to(dev).eval()
        for n in a.sizes or [5, 8, 10, 15]:
            cnt, mx = EVAL_PLAN.get(n, (10, 100 * n))
            acc, st = evaluate(model, n, cnt, mx, dev, 1000 + n)
            print(f"maze {2*n+1}x{2*n+1}: solved {acc*100:.0f}% ({cnt})  mean thinking steps {st:.1f}", flush=True)
