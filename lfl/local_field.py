"""Local Field Learner: backprop-free parts, taken unchanged from Local_Field_Learner.ipynb.

    HDEncoder     fixed random projection -> +1/-1 hyperdimensional code (no training, no tokenizer)
    LocalLayer    hidden layer trained from its own local target (Direct Random Target Projection)
    RidgeReadout  ridge regression kept as running sums A = F'F, C = F'Y: new data only adds, so it never forgets
    HashExperts   random-hyperplane routing to small experts that are grown on demand

Tests: python lfl/test_local_field.py
"""
import numpy as np

# ---------------------------------------------------------------
# Part 1 - universal encoder: fixed random projection -> bipolar HD vector
# No training, no tokenizer. Works on any numeric vector (sensor window,
# pixels, byte histogram...).
# ---------------------------------------------------------------
class HDEncoder:
    def __init__(self, d_in, D=4096, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_in, D)).astype(np.float32) / np.sqrt(d_in)
        self.b = rng.uniform(-1, 1, D).astype(np.float32)
        self.mu = None

    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        return self

    def __call__(self, X):
        Z = ((X - self.mu) / self.sd).astype(np.float32) @ self.W + self.b
        return np.sign(Z).astype(np.float32)          # +1 / -1 code


# ---------------------------------------------------------------
# Part 2 - local learning layer (Direct Random Target Projection).
# The layer updates from ITS OWN signal: the target projected through a
# fixed random matrix. No error is passed backward through other layers.
# ---------------------------------------------------------------
class LocalLayer:
    def __init__(self, d_in, d_out, n_classes, lr=0.01, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_in, d_out)).astype(np.float32) * np.sqrt(2 / d_in)
        self.B = rng.standard_normal((n_classes, d_out)).astype(np.float32)   # fixed, random
        self.lr = lr

    def __call__(self, X):
        return np.tanh(X @ self.W)

    def learn(self, X, Y, epochs=3, batch=128, seed=2):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
                H = self(X[i])
                target = np.tanh(Y[i] @ self.B)               # local target for this layer
                err = (H - target) * (1 - H ** 2)
                self.W -= self.lr * X[i].T @ err / len(i)
        return self


# ---------------------------------------------------------------
# Readout: ridge regression kept as running sums (A = F'F, C = F'Y).
# Adding new data only adds to the sums, so learning task 2 cannot
# overwrite task 1. Solving is one linear system - no epochs.
# ---------------------------------------------------------------
class RidgeReadout:
    def __init__(self, d, n_classes, lam=1.0):
        self.A = np.zeros((d, d), np.float64); self.C = np.zeros((d, n_classes), np.float64)
        self.lam = lam; self.Wo = None; self.n = 0

    def add(self, F, Y):
        self.A += F.T @ F; self.C += F.T @ Y; self.n += len(F); self.Wo = None

    def solve(self):
        d = self.A.shape[0]
        self.Wo = np.linalg.solve(self.A + self.lam * np.eye(d), self.C).astype(np.float32)

    def scores(self, F):
        if self.Wo is None: self.solve()
        return F @ self.Wo


# ---------------------------------------------------------------
# Parts 3 + 4 - hash-routed experts that grow on demand.
# A few random hyperplanes hash each input to a bucket; each bucket gets
# its own small readout, created the first time the bucket is seen
# ("grow instead of overwrite"). A shared global readout is the fallback.
# ---------------------------------------------------------------
class HashExperts:
    def __init__(self, D, n_classes, bits=4, d_expert=512, lam=1.0, min_n=40, seed=3):
        rng = np.random.default_rng(seed)
        self.H = rng.standard_normal((D, bits)).astype(np.float32)
        self.idx = rng.choice(D, d_expert, replace=False)      # each expert sees a slice
        self.k = n_classes; self.lam = lam; self.min_n = min_n
        self.glob = RidgeReadout(D, n_classes, lam)
        self.experts = {}

    def _bucket(self, F):
        bits = (F @ self.H > 0).astype(np.int64)
        return bits @ (1 << np.arange(bits.shape[1]))

    def add(self, F, Y):
        self.glob.add(F, Y)
        b = self._bucket(F)
        for k in np.unique(b):
            m = b == k
            if k not in self.experts:                          # grow a new expert
                self.experts[k] = RidgeReadout(len(self.idx), self.k, self.lam)
            self.experts[k].add(F[m][:, self.idx], Y[m])

    def scores(self, F):
        S = self.glob.scores(F)
        b = self._bucket(F)
        for k in np.unique(b):
            e = self.experts.get(k)
            if e is not None and e.n >= self.min_n:
                m = b == k
                S[m] = 0.5 * S[m] + 0.5 * e.scores(F[m][:, self.idx])
        return S

    @property
    def n_experts(self): return len(self.experts)


def onehot(y, k):
    Y = -np.ones((len(y), k), np.float32); Y[np.arange(len(y)), y] = 1; return Y
