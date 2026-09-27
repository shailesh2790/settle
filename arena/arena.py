"""The Arena — one scorecard for every idea.

Every entrant implements:
    name            str
    params          int
    train_flops     float   (approximate, from its own training log)
    solve(X) -> (pred_bool[H,W], info)   info = {"flops": float, "steps": int, "conf": float}
Tasks are generated with a difficulty knob; entrants are trained on easy instances and tested on harder ones.
Scorecard per difficulty: exact-solve rate, inference MFLOPs per instance, solves per GFLOP,
and calibration AUROC (does higher confidence mean more likely correct?).

    python arena.py 5 8        # run difficulties n=5 and n=8 (11x11, 17x17), append to arena_results.json
    python arena.py report     # print the leaderboard"""
import json, os, sys, time
import numpy as np
import settle as S
import ebm as E

# ------------------------------------------------------------------ tasks
def maze_task(n, rng):
    g, s, t, path = S.make_maze(n, rng)
    X = np.zeros((1, 2 * n + 1, 2 * n + 1, 3), np.float32)
    X[0, :, :, 0] = g; X[0, s[0], s[1], 1] = 1; X[0, t[0], t[1], 2] = 1
    return X, path > 0


TASKS = {"maze": {"make": maze_task, "train_range": "n = 4-7 (9x9 to 15x15)"}}


# ------------------------------------------------------------------ entrants
def conv_flops(px, cin, cout, k=3): return 2.0 * px * cin * cout * k * k


class SettleEntrant:
    name = "settle (recurrent latent)"

    def __init__(self, path="settle.json"):
        self.p, meta = S.load(path); C = self.p["C"]
        self.params = S.n_params(self.p)
        # training: ~2830 steps, avg batch ~26, avg 144 px, ~29 grad steps x3 (fwd+bwd) + ~20 no-grad steps
        per_step_px = 2 * ((C + 3) * C * 9 + C * C * 9)
        self.train_flops = 2830 * 26 * 144 * per_step_px * (29 * 3 + 20)

    def solve(self, X, max_iter=None, patience=4):
        C = self.p["C"]; H = X.shape[1]; px = H * H; max_iter = max_iter or 30 * H
        h, _ = S.encode(self.p, X); prev = None; stable = 0
        step_flops = conv_flops(px, C + 3, C) + conv_flops(px, C, C) + conv_flops(px, C, 1)
        for t in range(1, max_iter + 1):
            h, _, _ = S.step(self.p, h, X)
            logit = S.readout(self.p, h)[0][0, ..., 0]
            pred = (logit > 0) & (X[0, ..., 0] == 0)
            if prev is not None and np.array_equal(pred, prev): stable += 1
            else: stable = 0
            prev = pred
            if stable >= patience: break
        prob = 1 / (1 + np.exp(-logit)); openm = X[0, ..., 0] == 0
        undecided = int((((prob > 0.1) & (prob < 0.9)) & openm).sum())
        conf = -undecided - (50 if stable < patience else 0)
        return pred, {"flops": conv_flops(px, 3, C) + t * step_flops, "steps": t, "conf": conf, "_h": None}


class EBMEntrant:
    name = "energy (descent on answer)"

    def __init__(self, path="ebm.json"):
        self.p, meta = E.load(path); C = self.p["C"]
        self.params = E.n_params(self.p)
        fwd_px = 2 * (4 * C * 9 + C * C * 9 + C)
        steps = meta.get("steps", 6250)
        # per step: pos+neg fwd+bwd on 32 examples (~3x fwd each) + ~40 descent steps on 16 (~2x fwd each), ~144 px
        self.train_flops = steps * 144 * fwd_px * (2 * 32 * 3 + 40 * 16 * 2)

    def step_flops(self, px):
        C = self.p["C"]; return 2 * 2 * px * (4 * C * 9 + C * C * 9 + C)   # forward + backward-to-input

    def solve(self, X):
        H = X.shape[1]; px = H * H
        y, t, _ = E.think(self.p, X, 8 * H)
        conf = float(E.energy_confidence(self.p, X, y)[0])
        return (y[0, ..., 0] > 0.5) & (X[0, ..., 0] == 0), {"flops": t * self.step_flops(px), "steps": t, "conf": conf}


class SettleWithEnergyVerifier:
    """Generator from option 1, self-check from option 2: Settle answers, the energy scores the answer."""
    name = "settle + energy verifier"

    def __init__(self, settle, ebm):
        self.s, self.e = settle, ebm
        self.params = settle.params + ebm.params
        self.train_flops = settle.train_flops + ebm.train_flops

    def solve(self, X):
        pred, info = self.s.solve(X)
        y = pred[None, ..., None].astype(np.float32)
        conf = float(E.energy_confidence(self.e.p, X, y)[0])
        H = X.shape[1]
        return pred, {"flops": info["flops"] + self.e.step_flops(H * H) / 2, "steps": info["steps"], "conf": conf}


class SettleThinkMore:
    """Settle that spends extra compute ONLY when its own confidence says it is unsure.

    1. Think as usual (cap 30 x H steps, stop when the answer is stable).
    2. Confident (settled, no undecided cells)? Done: same cost as plain Settle.
    3. Unsure and hit the step cap? Keep thinking, up to `extend` x the cap.
    4. Still unsure? Search: copy the maze, wall off one of the `k` cells Settle is least sure of,
       and re-think each copy from scratch. Walling a cell that is NOT on the true path leaves the
       answer unchanged and often un-sticks the reasoner; walling a path cell makes the maze
       unsolvable, which shows up as low confidence. Settle's own confidence picks the winner
       (greedy, up to `rounds` rounds). No rule checker and no ground truth are used."""

    def __init__(self, settle, extend=4, k=4, rounds=3, search=True, require_path=False, name=None):
        self.s, self.p = settle, settle.p
        self.extend, self.k, self.rounds, self.search = extend, k, rounds, search
        self.params, self.train_flops = settle.params, settle.train_flops
        self.name = name or ("settle + think more when unsure" if search else "settle + longer only when unsure")
        self.require_path = require_path

    def _think(self, X, cap, patience=4):
        """Returns pred, prob, steps, settled, (undecided count). Resumable-free: always from scratch."""
        h, _ = S.encode(self.p, X); prev = None; stable = 0
        for t in range(1, cap + 1):
            h, _, _ = S.step(self.p, h, X)
            logit = S.readout(self.p, h)[0][0, ..., 0]
            pred = (logit > 0) & (X[0, ..., 0] == 0)
            stable = stable + 1 if prev is not None and np.array_equal(pred, prev) else 0
            prev = pred
            if stable >= patience: break
        prob = 1 / (1 + np.exp(-logit)); openm = X[0, ..., 0] == 0
        undecided = int((((prob > 0.1) & (prob < 0.9)) & openm).sum())
        return pred, prob, t, stable >= patience, undecided

    @staticmethod
    def _conf(settled, undecided): return -undecided - (0 if settled else 50)

    def _judge(self, settled, undecided, pred, X):
        """Confidence, and with require_path: a copy the model itself calls unsolvable never wins."""
        c = self._conf(settled, undecided)
        if self.require_path:
            s = tuple(np.argwhere(X[0, ..., 1] > 0)[0]); t = tuple(np.argwhere(X[0, ..., 2] > 0)[0])
            if not (pred[s] and pred[t]): c -= 1000
        return c

    def solve(self, X):
        C = self.p["C"]; H = X.shape[1]; px = H * H; cap = 30 * H
        step_flops = conv_flops(px, C + 3, C) + conv_flops(px, C, C) + conv_flops(px, C, 1)
        enc_flops = conv_flops(px, 3, C)
        pred, prob, t, settled, und = self._think(X, cap)
        flops, steps = enc_flops + t * step_flops, t
        if not settled:                                            # ran out of time: keep thinking
            pred, prob, t, settled, und = self._think(X, cap * self.extend)
            flops += enc_flops + t * step_flops; steps += t
        best = (self._judge(settled, und, pred, X), pred, prob, X)
        if self.search:
            for _ in range(self.rounds):
                conf, pred, prob, Xc = best
                if conf == 0: break                                # confident: stop spending
                openm = (Xc[0, ..., 0] == 0) & (Xc[0, ..., 1] == 0) & (Xc[0, ..., 2] == 0)
                unsure = np.where(openm, np.abs(prob - 0.5), np.inf).ravel()
                cands = [c for c in np.argsort(unsure)[: self.k] if np.isfinite(unsure[c])]
                improved = False
                for c in cands:
                    Xw = Xc.copy(); Xw[0, c // H, c % H, 0] = 1.0  # wall this cell off
                    pw, qw, tw, sw, uw = self._think(Xw, cap * self.extend)
                    flops += enc_flops + tw * step_flops; steps += tw
                    cw = self._judge(sw, uw, pw, Xw)
                    if cw > best[0]: best = (cw, pw, qw, Xw); improved = True
                if not improved: break
        conf, pred, _, _ = best
        return pred, {"flops": flops, "steps": steps, "conf": conf}


# ------------------------------------------------------------------ scoring
def auroc(conf, ok):
    """P(conf of a right answer > conf of a wrong one), ties counted as half. Ties are common
    (Settle's confidence is an integer), so ranks must not be broken arbitrarily."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, bool)
    if ok.all() or (~ok).all(): return None
    pos, neg = conf[ok][:, None], conf[~ok][None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def run(entrants, n, count, seed):
    rows = {}
    for ent in entrants:
        rng = np.random.default_rng(seed)                 # identical instances for every entrant
        ok, fl, st, cf = [], [], [], []
        t0 = time.time()
        for _ in range(count):
            X, Y = maze_task(n, rng)
            pred, info = ent.solve(X)
            ok.append(bool(np.array_equal(pred, Y))); fl.append(info["flops"]); st.append(info["steps"]); cf.append(info["conf"])
        rows[ent.name] = {"n": n, "size": f"{2*n+1}x{2*n+1}", "count": count, "solved": float(np.mean(ok)),
                          "mflops_per_instance": float(np.mean(fl) / 1e6), "steps": float(np.mean(st)),
                          "solves_per_gflop": float(np.sum(ok) / (np.sum(fl) / 1e9)), "auroc": auroc(cf, ok),
                          "params": ent.params, "train_pflops": ent.train_flops / 1e15, "secs": time.time() - t0}
    return rows


COUNTS = {5: 40, 8: 30, 10: 30, 15: 12, 20: 12}

if __name__ == "__main__":
    res_path = "arena_results.json"
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if sys.argv[1] == "report":
        for key in sorted(res, key=lambda k: int(k)):
            print(f"\n=== maze {res[key][next(iter(res[key]))]['size']}  (trained on 9x9-15x15) ===")
            print(f"{'entrant':<28} {'solved':>7} {'MFLOP/inst':>11} {'solves/GFLOP':>13} {'AUROC':>6} {'steps':>6}")
            for name, r in res[key].items():
                au = "  n/a" if r["auroc"] is None else f"{r['auroc']:.2f}"
                print(f"{name:<28} {r['solved']*100:>6.0f}% {r['mflops_per_instance']:>11.1f} {r['solves_per_gflop']:>13.2f} {au:>6} {r['steps']:>6.0f}")
        sys.exit()
    s, e = SettleEntrant(), EBMEntrant()
    ents = [s, e, SettleWithEnergyVerifier(s, e), SettleThinkMore(s, search=False), SettleThinkMore(s)]
    if os.path.exists("settle_nopath.json"):
        sn = SettleEntrant("settle_nopath.json"); sn.name = "settle, knows 'no path'"
        ents += [sn, SettleThinkMore(sn, search=False, name="settle, knows 'no path' + longer"),
                 SettleThinkMore(sn, require_path=True, name="settle, knows 'no path' + search")]
    only = os.environ.get("ARENA_ONLY")                    # e.g. ARENA_ONLY="settle" to run a subset by name
    if only: ents = [x for x in ents if only in x.name]
    for arg in sys.argv[1:]:
        n = int(arg); res[str(n)] = {**res.get(str(n), {}), **run(ents, n, COUNTS.get(n, 20), seed=1000 + n)}
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"done n={n}", {k: round(v['solved'], 2) for k, v in res[str(n)].items()}, flush=True)
