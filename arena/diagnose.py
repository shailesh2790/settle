"""Why does Settle fail on large mazes? Per failure: did it hit the step cap, how many cells are
undecided, how many are confidently wrong, and does a 4x longer budget fix it?

    python diagnose.py 15 20
"""
import sys
import numpy as np
import settle as S
from arena import maze_task, SettleEntrant, COUNTS


def think(p, X, max_iter, patience=4):
    h, _ = S.encode(p, X); prev = None; stable = 0
    for t in range(1, max_iter + 1):
        h, _, _ = S.step(p, h, X)
        logit = S.readout(p, h)[0][0, ..., 0]
        pred = (logit > 0) & (X[0, ..., 0] == 0)
        stable = stable + 1 if prev is not None and np.array_equal(pred, prev) else 0
        prev = pred
        if stable >= patience: break
    return pred, logit, t, stable >= patience


ent = SettleEntrant()
for arg in sys.argv[1:]:
    n = int(arg); H = 2 * n + 1; rng = np.random.default_rng(1000 + n); cnt = COUNTS.get(n, 12)
    print(f"\n=== {H}x{H}, {cnt} instances, default cap {30 * H} steps ===")
    for i in range(cnt):
        X, Y = maze_task(n, rng)
        pred, logit, t, settled = think(ent.p, X, 30 * H)
        ok = np.array_equal(pred, Y)
        openm = X[0, ..., 0] == 0; prob = 1 / (1 + np.exp(-logit))
        undecided = int((((prob > 0.1) & (prob < 0.9)) & openm).sum())
        sure_wrong = int(((prob >= 0.9) & ~Y & openm).sum() + ((prob <= 0.1) & Y & openm).sum())
        missing = int((Y & ~pred).sum()); extra = int((pred & ~Y).sum())
        line = f"#{i:<2} {'OK  ' if ok else 'FAIL'} steps {t:>5} {'settled' if settled else 'CAP    '} undecided {undecided:>3} sure-wrong {sure_wrong:>3} missing {missing:>3} extra {extra:>3} path {int(Y.sum()):>4}"
        if not ok:
            p4, _, t4, s4 = think(ent.p, X, 120 * H)
            line += f"   | 4x budget: {'OK' if np.array_equal(p4, Y) else 'fail'} at {t4} steps{'' if s4 else ' (cap)'}"
        print(line, flush=True)
