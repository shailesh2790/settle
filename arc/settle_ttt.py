"""Settle core with test-time training on ARC tasks: learn the rule from the task's own 2-5 examples.

For each task, a small recurrent grid network (same idea as the maze reasoner: encode once, then apply ONE
shared update repeatedly with the problem re-injected, read out every cell in parallel) is trained from
scratch only on that task's examples, with a loss after every thinking step (deep supervision) and the update
initialised near "do nothing". No augmentation by default: many ARC rules are about specific colours or
directions, so recolouring or rotating the examples changes the rule. The two attempts come from two seeds.
CPU only by default.

    python arc/settle_ttt.py --set dev [--limit 40]
"""
import argparse, json, os, sys, time
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU only: the GPU is reserved for paused training
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_lab import load_set, score, D8, ROOT

INV = [lambda g: g, lambda g: np.rot90(g, -1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1),
       lambda g: g.T, lambda g: np.fliplr(g), lambda g: np.flipud(g), lambda g: np.rot90(g, 2).T]


class SettleGrid(nn.Module):
    def __init__(self, C=48):
        super().__init__()
        self.enc = nn.Conv2d(11, C, 3, padding=1)            # 10 colours + "inside the grid" channel
        self.c1 = nn.Conv2d(C + 11, C, 3, padding=1)
        self.c2 = nn.Conv2d(C, C, 3, padding=1)
        self.out = nn.Conv2d(C, 10, 1)
        with torch.no_grad(): self.c2.weight.mul_(0.1); self.c2.bias.zero_()   # start near "do nothing"

    def forward(self, x, steps, all_steps=False):
        h = torch.relu(self.enc(x)); outs = []
        for _ in range(steps):                               # same weights every step, input re-injected
            a = torch.relu(self.c1(torch.cat([h, x], 1)))
            h = torch.tanh(h + self.c2(a)); outs.append(self.out(h))
        return outs if all_steps else outs[-1]


def onehot(grids, H, W):
    """Pad each grid to HxW; channel 10 marks real cells."""
    X = np.zeros((len(grids), 11, H, W), np.float32)
    for i, g in enumerate(grids):
        h, w = g.shape
        X[i, g, np.arange(h)[:, None], np.arange(w)[None, :]] = 1.0
        X[i, 10, :h, :w] = 1.0
    return torch.from_numpy(X)


def augment(pairs, rng, n):
    """n augmented copies: random rotation/flip plus a colour permutation that keeps black (0) fixed."""
    out = []
    for _ in range(n):
        g = D8[rng.integers(8)]; perm = np.concatenate([[0], rng.permutation(9) + 1])
        out += [(perm[g(x)], perm[g(y)]) for x, y in pairs]
    return out


def fit_one(task, steps, iters, C, seed, lr=3e-3, aug=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    H = W = 15; base = task["train"]
    model = SettleGrid(C); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        batch = base + (augment(base, rng, 3) if aug else [])
        X = onehot([x for x, _ in batch], H, W)
        Y = torch.full((len(batch), H, W), -100, dtype=torch.long)
        for i, (_, y) in enumerate(batch): Y[i, :y.shape[0], :y.shape[1]] = torch.from_numpy(y)
        outs = model(X, steps, all_steps=True)              # deep supervision: a loss after every step
        loss = sum(F.cross_entropy(o, Y, ignore_index=-100) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = lambda x: model(onehot([x], H, W), steps)[0, :, :x.shape[0], :x.shape[1]].argmax(0).numpy()
        fit = float(np.mean([np.array_equal(pred(x), y) for x, y in base]))   # does it reproduce its own examples?
        return [pred(x) for x, _ in task["test"]], fit, loss.item()


def solve_task(task, steps=8, iters=300, C=48, seeds=(0, 1)):
    """Two seeds give the two attempts. A seed that reproduces all its examples is tried first."""
    runs = [fit_one(task, steps, iters, C, s) for s in seeds]
    runs.sort(key=lambda r: -r[1])
    attempts = [[r[0][i] for r in runs] for i in range(len(task["test"]))]
    return attempts, max(r[1] for r in runs), min(r[2] for r in runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "heldout"]); ap.add_argument("--limit", type=int)
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--steps", type=int, default=8); ap.add_argument("--C", type=int, default=48)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    tasks = load_set(a.set)[: a.limit]
    out = a.out or os.path.join(ROOT, "runs", "arc", f"settle_ttt_{a.set}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["task"] for l in open(out)} if os.path.exists(out) else set()
    t0 = time.time()
    with open(out, "a") as log:
        for i, t in enumerate(tasks):
            if t["id"] in done: continue
            ts = time.time(); attempts, fit, loss = solve_task(t, a.steps, a.iters, a.C)
            ok = score(t, attempts)
            log.write(json.dumps({"task": t["id"], "solved": ok, "fits_examples": fit, "final_loss": round(loss, 4),
                                  "secs": round(time.time() - ts, 1), "iters": a.iters, "steps": a.steps, "C": a.C}) + "\n"); log.flush()
            print(f"{i + 1}/{len(tasks)} {t['id']} {'SOLVED' if ok else 'no':<6} fits {fit:.2f} {time.time() - ts:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(out)]
    n = len(rows); s = sum(r["solved"] for r in rows); f = sum(r["fits_examples"] == 1.0 for r in rows)
    print(f"\n{a.set}: solved {s}/{n} ({s / n * 100:.1f}%) | fits all its examples on {f}/{n} | solved when it fits: {sum(r['solved'] for r in rows if r['fits_examples'] == 1.0)}/{f}")
