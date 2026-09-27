"""Reference outputs from the PyTorch FactorSettle for tests/sudoku_parity.test.js, plus the
sample puzzles the site offers (site/data/sudoku_samples.json)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import settle_graph as G
from tasks import sudoku as task

root = os.path.join(os.path.dirname(__file__), "..")
ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "runs", "factor_10min.pt")
model, _ = G.load_model(ckpt, "cpu")
q, a, r = task.load("test")
rng = np.random.default_rng(11)

cases = []
easy = np.nonzero(r == 0)[0]
for i in list(rng.choice(len(q), 2, replace=False)) + list(rng.choice(easy, 3, replace=False)):
    xq = torch.from_numpy(q[i:i + 1].astype(np.int64))
    with torch.no_grad():
        h, x = model.encode(xq)
        for _ in range(16): h = model.step(h, x)
        lg16 = model.readout(h)[0]
        pred, st, ok = G.solve_batch(model, xq, max_steps=128, check_every=1)
    cases.append({"q": q[i].tolist(), "logits16": lg16.flatten().tolist(), "ok": bool(ok[0]), "steps": int(st[0]), "pred": pred[0].tolist()})
json.dump(cases, open(os.path.join(root, "tests", "sudoku_fixture.json"), "w"))

samples = []
for lo, hi, name in task.RATING_BANDS:
    idx = np.nonzero((r >= lo) & (r <= hi))[0]
    for i in rng.choice(idx, 50, replace=False):
        samples.append({"q": "".join(map(str, q[i])), "a": "".join(map(str, a[i])), "rating": int(r[i]), "band": name})
os.makedirs(os.path.join(root, "site", "data"), exist_ok=True)
json.dump(samples, open(os.path.join(root, "site", "data", "sudoku_samples.json"), "w"), separators=(",", ":"))
print(f"{len(cases)} parity cases ({sum(c['ok'] for c in cases)} solved by thinking), {len(samples)} samples")
