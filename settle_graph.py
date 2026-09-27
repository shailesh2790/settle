"""
Settle on constraint graphs: a weight-tied recurrent reasoner for problems where variables constrain each other.

Same idea as settle.py (iterate in latent space, re-inject the problem every step, same weights
every step), but cells talk along the edges of a constraint graph instead of through 3x3 convolutions:

    h0     = embed(x)
    e_ij   = relu(A h_i + B h_j + E type_ij)          message along each edge j -> i
    h_t+1  = norm(h_t + MLP([h_t, embed(x), W sum_j e_ij]))
    answer = softmax(out(h_T))                         every variable answered in parallel

Training uses deep supervision on a rolling batch (as in HRM/TRM): every optimizer step runs K
thinking steps on the current batch, the latent state is carried over (detached) to the next
optimizer step, and puzzles leave the batch once solved, making room for fresh ones.

Because a constraint problem's answer can be checked, inference halts when the answer VERIFIES,
not just when it stops changing.

    python settle_graph.py train --minutes 60             # train on Sudoku-Extreme
    python settle_graph.py eval --n 2000                   # held-out accuracy by difficulty
    python settle_graph.py solve ".9...12...3..284.6..."   # solve one puzzle
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks import sudoku as task


class GraphSettle(nn.Module):
    def __init__(self, n_classes, edges, n_edge_types, D=128, H=128):
        super().__init__()
        dst, src, etype = edges
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("etype", torch.as_tensor(etype, dtype=torch.float32))
        n_nodes = int(self.dst.max()) + 1
        deg = torch.bincount(self.dst, minlength=n_nodes)
        # Fast path when every node has the same in-degree and edges are grouped by destination.
        self.uniform = bool((deg == deg[0]).all() and (self.dst == torch.arange(n_nodes).repeat_interleave(deg[0])).all())
        self.register_buffer("deg", deg.float().clamp(min=1)[:, None])
        self.D, self.H, self.n_classes = D, H, n_classes
        self.embed = nn.Embedding(n_classes + 1, D)          # 0 = unknown, 1..n = given value
        self.h_init = nn.Parameter(torch.zeros(D))
        self.msg_dst = nn.Linear(D, H)
        self.msg_src = nn.Linear(D, H, bias=False)
        self.msg_type = nn.Linear(n_edge_types, H, bias=False)
        self.msg_out = nn.Linear(H, D)
        self.upd = nn.Sequential(nn.Linear(3 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, n_classes)

    def encode(self, xq):
        x = self.embed(xq)
        return self.norm(x + self.h_init), x

    def step(self, h, x):
        e = F.relu(self.msg_dst(h)[:, self.dst] + self.msg_src(h)[:, self.src] + self.msg_type(self.etype))
        if self.uniform:
            m = e.view(e.shape[0], h.shape[1], -1, self.H).sum(2)
        else:
            m = torch.zeros(h.shape[0], h.shape[1], self.H, device=h.device, dtype=e.dtype).index_add_(1, self.dst, e)
        m = self.msg_out(m / self.deg)
        return self.norm(h + self.upd(torch.cat([h, x, m], -1)))

    def readout(self, h): return self.out(h)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


class FactorSettle(nn.Module):
    """Variables talk through the constraint groups they belong to (rows, columns, boxes, ...).

        g_v      = P h_v                                   what v contributes to its groups
        rest_gv  = sum_{u in g} g_u  -  g_v                 the rest of group g, seen from v
        m_gv     = relu(R rest_gv + S h_v + type_g)        message from group g to v
        h_v     <- norm(h_v + MLP([h_v, x_v, W concat_types(mean_g m_gv)]))

    Excluding v from its own group sum is what lets it express all-different reasoning
    ("no other cell in this row can still be a 7") with a handful of weights.
    """
    def __init__(self, n_classes, membership, n_group_types, D=128, H=128):
        super().__init__()
        var, grp, typ = (torch.as_tensor(m, dtype=torch.long) for m in membership)
        self.register_buffer("var", var); self.register_buffer("grp", grp); self.register_buffer("typ", typ)
        self.n_vars, self.n_groups, self.T = int(var.max()) + 1, int(grp.max()) + 1, n_group_types
        slot = var * n_group_types + typ                                   # (variable, group type) bucket
        self.register_buffer("slot", slot)
        self.register_buffer("slot_n", torch.bincount(slot, minlength=self.n_vars * self.T).float().clamp(min=1)[:, None])
        self.D, self.H, self.n_classes = D, H, n_classes
        self.embed = nn.Embedding(n_classes + 1, D)
        self.h_init = nn.Parameter(torch.zeros(D))
        self.pool = nn.Linear(D, H, bias=False)
        self.rest = nn.Linear(H, H)
        self.self_ = nn.Linear(D, H, bias=False)
        self.type_emb = nn.Embedding(n_group_types, H)
        self.msg_out = nn.Linear(n_group_types * H, D)
        self.upd = nn.Sequential(nn.Linear(3 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, n_classes)

    def encode(self, xq):
        x = self.embed(xq)
        return self.norm(x + self.h_init), x

    def step(self, h, x):
        B = h.shape[0]
        g = self.pool(h)[:, self.var]                                                   # B, M, H
        S = torch.zeros(B, self.n_groups, self.H, device=h.device, dtype=g.dtype).index_add_(1, self.grp, g)
        m = F.relu(self.rest(S[:, self.grp] - g) + self.self_(h)[:, self.var] + self.type_emb(self.typ))
        agg = torch.zeros(B, self.n_vars * self.T, self.H, device=h.device, dtype=m.dtype).index_add_(1, self.slot, m)
        agg = (agg / self.slot_n).view(B, self.n_vars, self.T * self.H)
        return self.norm(h + self.upd(torch.cat([h, x, self.msg_out(agg)], -1)))

    def readout(self, h): return self.out(h)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())


def make_model(cfg):
    if cfg.get("arch", "pairwise") == "factor":
        return FactorSettle(task.N_CLASSES, task.MEMBERSHIP, task.N_GROUP_TYPES, D=cfg["D"], H=cfg["H"])
    return GraphSettle(task.N_CLASSES, task.EDGES, task.N_EDGE_TYPES, D=cfg["D"], H=cfg["H"])


def predict(logits, xq):
    """Most likely value per cell, with givens kept as given. Returns digits 1..n."""
    p = logits.argmax(-1) + 1
    return torch.where(xq > 0, xq, p)


@torch.no_grad()
def solve_batch(model, xq, max_steps=256, check_every=4, return_probs=False):
    """Think until each puzzle's answer verifies (or max_steps). Returns preds, steps used, solved mask
    (and, with return_probs, each puzzle's last per-cell class probabilities)."""
    h, x = model.encode(xq)
    B = xq.shape[0]; dev = xq.device
    steps = torch.full((B,), max_steps, dtype=torch.long); solved = torch.zeros(B, dtype=torch.bool)
    final = torch.zeros_like(xq)
    probs = torch.zeros(B, xq.shape[1], model.n_classes, device=dev) if return_probs else None
    active = torch.arange(B, device=dev)
    q_np = xq.cpu().numpy()
    for t in range(1, max_steps + 1):
        h = model.step(h, x)
        if t % check_every and t != max_steps: continue
        logits = model.readout(h).float()
        pred = predict(logits, xq[active])
        if return_probs: probs[active] = logits.softmax(-1)
        ok = task.is_solved(q_np[active.cpu().numpy()], pred.cpu().numpy())
        final[active] = pred
        idx = active.cpu()[torch.from_numpy(ok)]
        solved[idx] = True; steps[idx] = t
        if ok.all(): break
        keep = torch.from_numpy(~ok).to(dev)
        active, h, x = active[keep], h[keep], x[keep]
    return (final, steps, solved, probs) if return_probs else (final, steps, solved)


def peers_conflict(q):
    """True where a partially filled grid already breaks a constraint (duplicate given in a unit)."""
    u = np.sort(q[:, task.UNITS], axis=2)
    return ((u[:, :, 1:] == u[:, :, :-1]) & (u[:, :, 1:] > 0)).any((1, 2))


@torch.no_grad()
def search(model, q, beam=32, branch=3, settle_steps=48, max_rounds=40):
    """Verified search for one puzzle (uint8 [81], 0 = blank).

    Round 0 is plain thinking. If the answer does not verify, every surviving state branches on the
    blank cell the model is least sure about (lowest top probability), trying its `branch` most
    likely values as new givens. Children that break a constraint are dropped; the `beam` states
    with the highest cumulative log-probability survive. All states in a round think together as
    one batch. Any returned answer has passed the checker, so it is correct by construction.
    """
    dev = next(model.parameters()).device
    states, scores = q[None].astype(np.int64), np.zeros(1)
    total_steps = 0
    for rnd in range(max_rounds + 1):
        xq = torch.from_numpy(states).to(dev)
        pred, st, ok, P = solve_batch(model, xq, max_steps=settle_steps, return_probs=True)
        total_steps += int(st.sum())
        if ok.any():
            i = int(ok.nonzero()[0])
            return pred[i].cpu().numpy(), {"rounds": rnd, "states": len(states), "thinking_steps": total_steps}
        P = P.cpu().numpy(); kids, kid_scores = [], []
        for s, sc, p in zip(states, scores, P):
            top = p.max(-1)
            top[s > 0] = 2.0                                        # givens are not branch candidates
            cell = int(top.argmin())
            if top[cell] >= 2.0: continue                          # fully filled but wrong: dead end
            for v in np.argsort(-p[cell])[:branch]:
                if p[cell, v] < 1e-4: break
                c = s.copy(); c[cell] = v + 1
                kids.append(c); kid_scores.append(sc + np.log(p[cell, v] + 1e-9))
        if not kids: break
        kids, kid_scores = np.array(kids), np.array(kid_scores)
        alive = ~peers_conflict(kids)
        kids, kid_scores = kids[alive], kid_scores[alive]
        if not len(kids): break
        order = np.argsort(-kid_scores)[:beam]
        states, scores = kids[order], kid_scores[order]
    return None, {"rounds": rnd, "states": len(states), "thinking_steps": total_steps}


# ---------------------------------------------------------------- training
class RollingBatch:
    """B puzzle slots. Each slot keeps its latent state across optimizer steps until the puzzle is
    solved or has had `max_segments` segments, then it is refilled with a fresh augmented puzzle."""
    def __init__(self, q, a, B, max_segments, rng, dev):
        self.q, self.a, self.B, self.max_seg, self.rng, self.dev = q, a, B, max_segments, rng, dev
        self.xq = torch.zeros(B, 81, dtype=torch.long, device=dev); self.y = torch.zeros_like(self.xq)
        self.h = None; self.seg = torch.zeros(B, dtype=torch.long, device=dev)
        self.fresh = torch.ones(B, dtype=torch.bool, device=dev); self.served = 0

    def refill(self, model, replace):
        n = int(replace.sum())
        if n:
            i = self.rng.integers(len(self.q), size=n)
            qq, aa = task.augment(self.q[i], self.a[i], self.rng)
            self.xq[replace] = torch.from_numpy(qq.astype(np.int64)).to(self.dev)
            self.y[replace] = torch.from_numpy(aa.astype(np.int64) - 1).to(self.dev)
            self.seg[replace] = 0; self.served += n
        return model.encode(self.xq)


def on_battery():
    """True when a Windows laptop is running on battery (GPUs throttle hard and the run would drain it)."""
    if os.name != "nt": return False
    import ctypes
    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte), ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))) and s.ACLineStatus == 0


def keep_awake():
    """Stop Windows from sleeping while this process runs. Reverts automatically when it exits."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


EXIT_PAUSED_FOR_POWER = 3


def train(a):
    """Laptop-friendly training. `--minutes` is the total training budget across restarts: a resumed
    run continues the same learning-rate schedule. Exits with code 3 (after checkpointing) when the
    laptop is unplugged, and stops cleanly when a file named STOP appears next to the checkpoint."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"D": a.D, "H": a.H, "K": a.K, "task": "sudoku", "arch": a.arch}
    model = make_model(cfg).to(dev)
    ema = make_model(cfg).to(dev); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    step, log, trained, best = 0, [], 0.0, -1.0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev)
        cfg = ck["cfg"]
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        step, log, trained, best = ck["step"], ck.get("log", []), ck.get("trained_secs", 0.0), ck.get("best_val", -1.0)
    torch.manual_seed(a.seed + step); rng = np.random.default_rng(a.seed + step)
    q, ans, _ = task.load("train")
    qv, av, rv = task.load("test", limit=None)
    vi = np.random.default_rng(1).choice(len(qv), 512, replace=False)          # fixed quick-val subset
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)
    stop_file = os.path.join(os.path.dirname(os.path.abspath(a.out)), "STOP")
    best_path = a.out.replace(".pt", "_best.pt")
    budget = a.minutes * 60
    keep_awake()
    print(f"device {dev}  params {model.n_params:,}  D={cfg['D']} H={cfg['H']} K={a.K}  batch {a.batch}  "
          f"budget {a.minutes} min, {trained / 60:.1f} done{'  (resumed at step %d)' % step if step else ''}", flush=True)

    def save(path=a.out):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "step": step,
                    "log": log, "trained_secs": trained, "budget_min": a.minutes, "best_val": best}, path)

    rb = RollingBatch(q, ans, a.batch, a.max_segments, rng, dev)
    replace = torch.ones(a.batch, dtype=torch.bool, device=dev)
    start_step, last_log, last_power, tick = step, time.time(), time.time(), time.time()
    while trained < budget:
        now = time.time(); trained += min(now - tick, 5.0); tick = now      # a sleep/stall never counts as training
        if now - last_power > 30:
            last_power = now
            if os.path.exists(stop_file):
                save(); print(f"STOP file found: saved step {step}, stopping.", flush=True); return
            if on_battery():
                save(); print(f"on battery: saved step {step}, pausing until plugged in.", flush=True); sys.exit(EXIT_PAUSED_FOR_POWER)
        frac = min(1.0, trained / budget)
        warm = min(1.0, (step + 1) / a.warmup) if start_step == 0 else 1.0
        lr = a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups: g["lr"] = lr
        with torch.autocast(**amp):
            h0, x = rb.refill(model, replace)
            h = h0 if rb.h is None else torch.where(replace[:, None, None], h0, rb.h.to(h0.dtype))
            for _ in range(a.K): h = model.step(h, x)
            logits = model.readout(h).float()
        loss = F.cross_entropy(logits.reshape(-1, task.N_CLASSES), rb.y.reshape(-1), reduction="none").view(a.batch, -1)
        blank = (rb.xq == 0).float()
        loss = ((loss * blank).sum(1) / blank.sum(1).clamp(min=1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - a.ema)
            rb.h = h.detach(); rb.seg += 1
            cell_ok = (logits.argmax(-1) == rb.y) | (rb.xq > 0)
            done = cell_ok.all(1)
            replace = done | (rb.seg >= a.max_segments)
        if time.time() - last_log > a.log_secs:
            last_log = time.time()
            ema.eval()
            with torch.autocast(**amp):
                pred, st, ok = solve_batch(ema, torch.from_numpy(qv[vi].astype(np.int64)).to(dev), max_steps=a.val_steps)
            val = ok.float().mean().item(); mins = trained / 60
            log.append([step, round(mins, 2), round(loss.item(), 4), round(val, 4)])
            improved = val > best
            if improved: best = val
            print(f"{mins:6.1f} min  step {step:6d}  loss {loss.item():.4f}  batch cells {cell_ok.float().mean():.3f}  "
                  f"puzzles seen {rb.served:,}  val solved {val*100:5.1f}%{' (best)' if improved else ''}  lr {lr:.1e}", flush=True)
            save()
            if improved: save(best_path)
            tick = time.time()                                       # validation time is not training time
    save()
    print(f"finished: {trained / 60:.1f} min trained, step {step}, best val {best * 100:.1f}% -> {best_path}", flush=True)


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    m = make_model(ck["cfg"]).to(dev); m.load_state_dict(ck["ema"]); m.eval()
    return m, ck


def evaluate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.ckpt, dev)
    q, ans, r = task.load("test")
    idx = np.random.default_rng(a.seed).choice(len(q), a.n, replace=False)
    q, ans, r = q[idx], ans[idx], r[idx]
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)
    solved, steps = np.zeros(a.n, bool), np.zeros(a.n, int); t0 = time.time()
    for i in range(0, a.n, a.bs):
        xq = torch.from_numpy(q[i:i + a.bs].astype(np.int64)).to(dev)
        with torch.autocast(**amp):
            pred, st, ok = solve_batch(model, xq, max_steps=a.max_steps)
        solved[i:i + a.bs] = ok.numpy(); steps[i:i + a.bs] = st.numpy()
        exact = (pred.cpu().numpy() == ans[i:i + a.bs]).all(1)
        assert (exact == ok.numpy()).all(), "verified-solved must equal exact match (solutions are unique)"
    secs = time.time() - t0
    searched = solved.copy(); s_secs = 0.0
    if a.search:
        t1 = time.time()
        for i in np.nonzero(~solved)[0]:
            sol, info = search(model, q[i], beam=a.beam, settle_steps=a.settle_steps, max_rounds=a.max_rounds)
            if sol is not None:
                assert (sol == ans[i]).all(); searched[i] = True
        s_secs = time.time() - t1
    print(f"model {a.ckpt}: {model.n_params:,} params, trained {ck['step']} steps")
    print(f"Sudoku-Extreme test, {a.n} random puzzles, up to {a.max_steps} thinking steps, {secs / a.n * 1000:.1f} ms/puzzle on {dev}")
    print(f"  thinking only: solved {solved.mean() * 100:.1f}%   mean steps when solved {steps[solved].mean() if solved.any() else float('nan'):.0f}")
    if a.search:
        print(f"  + verified search (beam {a.beam}): solved {searched.mean() * 100:.1f}%   "
              f"{s_secs / max(1, (~solved).sum()):.2f} s per searched puzzle")
    print(f"  {'difficulty':>18}  {'count':>6}  {'thinking':>8}" + ("  {:>8}".format("+search") if a.search else ""))
    for lo, hi, name in task.RATING_BANDS:
        m = (r >= lo) & (r <= hi)
        if m.any():
            print(f"  {name:>18}  {m.sum():>6}  {solved[m].mean() * 100:7.1f}%" + (f"  {searched[m].mean() * 100:7.1f}%" if a.search else ""))
    return searched.mean()


def solve_one(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.ckpt, dev)
    s = a.puzzle.replace(".", "0").replace(" ", "")
    assert len(s) == 81 and s.isdigit(), "give 81 characters, digits with . or 0 for blanks"
    xq = torch.tensor([[int(c) for c in s]], device=dev)
    t0 = time.time(); pred, st, ok = solve_batch(model, xq, max_steps=a.max_steps, check_every=1)
    note = ""
    if not ok[0] and a.search:
        sol, info = search(model, xq[0].cpu().numpy(), beam=a.beam, settle_steps=a.settle_steps, max_rounds=a.max_rounds)
        if sol is not None:
            pred[0] = torch.from_numpy(sol); ok[0] = True
            note = f" + search ({info['rounds']} branching rounds, {info['thinking_steps']} batched steps)"
        else:
            note = f" + search gave up after {info['rounds']} rounds"
    print(task.show(xq[0].cpu()), "\n")
    print(task.show(pred[0].cpu()))
    v = int(task.violations(xq.cpu().numpy(), pred.cpu().numpy())[0])
    print(f"\n{'verified solution' if ok[0] else f'NOT solved ({v} constraint violations)'} after {int(st[0])} thinking steps{note}, {time.time() - t0:.2f}s")


def export(a):
    """Write <out_dir>/sudoku_model.json (architecture, tensor index, provenance) + sudoku_model.bin
    (little-endian float32, EMA weights) for the browser engine in site/sudoku-engine.js."""
    model, ck = load_model(a.ckpt, "cpu")
    cfg = ck["cfg"]
    assert cfg.get("arch") == "factor", "the browser engine implements the factor architecture"
    os.makedirs(a.out_dir, exist_ok=True)
    tensors, blobs, offset = {}, [], 0
    for name, t in model.state_dict().items():
        if name in ("var", "grp", "typ", "slot", "slot_n"): continue          # rebuilt from the task in JS
        arr = t.detach().float().contiguous().numpy().astype("<f4")
        tensors[name] = {"shape": list(arr.shape), "offset": offset}
        blobs.append(arr.tobytes()); offset += arr.size
    log = ck.get("log", [])
    meta = {"arch": "factor", "D": cfg["D"], "H": cfg["H"], "n_classes": task.N_CLASSES, "n_group_types": task.N_GROUP_TYPES,
            "params": model.n_params, "train_steps": ck["step"], "train_minutes": log[-1][1] if log else None,
            "val_solved": log[-1][3] if log else None, "exported": time.strftime("%Y-%m-%d %H:%M"),
            "checkpoint": os.path.basename(a.ckpt), "tensors": tensors}
    json.dump(meta, open(os.path.join(a.out_dir, "sudoku_model.json"), "w"), indent=1)
    open(os.path.join(a.out_dir, "sudoku_model.bin"), "wb").write(b"".join(blobs))
    print(f"exported {model.n_params:,} params ({offset * 4 / 1024:.0f} KB) from {a.ckpt} step {ck['step']} to {a.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "solve", "export"])
    ap.add_argument("puzzle", nargs="?")
    ap.add_argument("--arch", choices=["factor", "pairwise"], default="factor")
    ap.add_argument("--D", type=int, default=128); ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--K", type=int, default=8, help="thinking steps per optimizer step")
    ap.add_argument("--max-segments", type=int, default=16, help="optimizer steps a puzzle stays in the batch")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--log-secs", type=float, default=120); ap.add_argument("--val-steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="settle_sudoku.pt"); ap.add_argument("--ckpt", default="settle_sudoku.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--search", action="store_true", help="verified beam search when thinking alone fails")
    ap.add_argument("--beam", type=int, default=32); ap.add_argument("--settle-steps", type=int, default=48)
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--out-dir", default="site/models")
    a = ap.parse_intermixed_args()
    {"train": train, "eval": evaluate, "solve": solve_one, "export": export}[a.cmd](a)
