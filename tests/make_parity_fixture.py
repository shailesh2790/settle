"""Dump NumPy reference outputs so the JS engine (web/settle.js) can be checked against them."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import settle

root = os.path.join(os.path.dirname(__file__), "..")
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "settle_weights.json"          # e.g. arena/settle_nopath.json
FIXTURE = sys.argv[2] if len(sys.argv) > 2 else "parity_fixture.json"
p, _ = settle.load(os.path.join(root, WEIGHTS))
rng = np.random.default_rng(7); cases = []
for n in (5, 8, 10):
    X, Y, M = settle.batch(n, 1, rng)
    h, _ = settle.encode(p, X)
    for _ in range(10): h, _, _ = settle.step(p, h, X)
    logit10 = settle.readout(p, h)[0]
    pred, t, _ = settle.solve(p, X, max_iter=300)
    cases.append({"S": 2 * n + 1, "X": X[0].ravel().tolist(), "logit10": logit10.ravel().tolist(),
                  "pred": pred.ravel().astype(int).tolist(), "steps": t})
json.dump(cases, open(os.path.join(os.path.dirname(__file__), FIXTURE), "w"))
print("wrote", len(cases), "cases")
