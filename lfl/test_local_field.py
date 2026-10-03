"""python lfl/test_local_field.py  -> checks the properties the project relies on."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfl.local_field import HDEncoder, RidgeReadout, HashExperts, LocalLayer, onehot

rng = np.random.default_rng(0)
X = rng.standard_normal((600, 20)).astype(np.float32); y = (X[:, 0] + X[:, 1] > 0).astype(int) + 2 * (X[:, 2] > 0)
enc = HDEncoder(20, D=512).fit(X); F, Y = enc(X), onehot(y, 4)

# 1. the code is +1/-1 and deterministic
assert set(np.unique(F)) <= {-1.0, 1.0} and np.array_equal(F, HDEncoder(20, D=512).fit(X)(X))

# 2. no forgetting: adding data in chunks gives the same readout as adding it all at once
r_all = RidgeReadout(512, 4); r_all.add(F, Y); r_all.solve()
r_seq = RidgeReadout(512, 4)
for part in np.array_split(np.arange(len(F)), 5): r_seq.add(F[part], Y[part])
r_seq.solve()
assert np.allclose(r_all.Wo, r_seq.Wo, atol=1e-4), "sequential and joint readouts must match"

# 3. it actually learns (well above chance on held-out data)
Xt = rng.standard_normal((300, 20)).astype(np.float32); yt = (Xt[:, 0] + Xt[:, 1] > 0).astype(int) + 2 * (Xt[:, 2] > 0)
r100 = RidgeReadout(512, 4, lam=100); r100.add(F, Y)            # few samples per feature: regularise strongly
acc = (r100.scores(enc(Xt)).argmax(1) == yt).mean(); assert acc > 0.6, acc

# 4. experts grow on demand and still score every input
he = HashExperts(512, 4, bits=3, min_n=10); he.add(F, Y)
assert he.n_experts >= 2 and he.scores(enc(Xt)).shape == (300, 4)

# 5. the local layer trains without backprop through other layers and keeps shapes
L = LocalLayer(512, 64, 4, lr=0.002).learn(F, Y, epochs=1); assert L(F).shape == (600, 64)
print(f"all local field learner tests passed (held-out accuracy {acc:.2f}, {he.n_experts} experts)")
