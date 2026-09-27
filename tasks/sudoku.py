"""
Sudoku as a constraint graph for settle_graph.py.

The model is told WHICH cells interact (edges, typed row / column / box) but never the rule
(all-different). It has to learn what the interaction means from solved examples.

Data: Sudoku-Extreme (github.com/sapientinc/HRM), the benchmark used by HRM and TRM.
    data/sudoku_extreme_train_part.csv   first ~533k rows of train.csv
    data/sudoku_extreme_test.csv         423k held-out puzzles, inequivalent to train
"""
import os
import numpy as np

N_NODES, N_CLASSES, N_EDGE_TYPES = 81, 9, 3
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def build_edges():
    """Directed edges (dst, src) between every pair of peers, with a 3-bit type: same row/col/box."""
    dst, src, typ = [], [], []
    for i in range(81):
        ri, ci = divmod(i, 9); bi = (ri // 3) * 3 + ci // 3
        for j in range(81):
            if i == j: continue
            rj, cj = divmod(j, 9); bj = (rj // 3) * 3 + cj // 3
            t = [ri == rj, ci == cj, bi == bj]
            if any(t): dst.append(i); src.append(j); typ.append(t)
    return np.array(dst), np.array(src), np.array(typ, np.float32)


EDGES = build_edges()                                  # 81 * 20 = 1620 edges

UNITS = np.array([[r * 9 + c for c in range(9)] for r in range(9)] +
                 [[r * 9 + c for r in range(9)] for c in range(9)] +
                 [[(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)])


def load(split, limit=None):
    """Returns (puzzles uint8 [N,81] with 0 = blank, solutions uint8 [N,81] digits 1-9, rating int32 [N])."""
    name = {"train": "sudoku_extreme_train_part", "test": "sudoku_extreme_test"}[split]
    cache = os.path.join(ROOT, "data", name + ".npz")
    if not os.path.exists(cache):
        q, a, r = [], [], []
        with open(os.path.join(ROOT, "data", name + ".csv")) as f:
            next(f)
            for line in f:
                p = line.strip().split(",")
                if len(p) != 4 or len(p[1]) != 81 or len(p[2]) != 81: continue   # truncated tail line
                q.append(p[1].replace(".", "0")); a.append(p[2]); r.append(int(p[3]))
        to = lambda s: np.frombuffer("".join(s).encode(), np.uint8).reshape(-1, 81) - ord("0")
        np.savez_compressed(cache, q=to(q), a=to(a), r=np.array(r, np.int32))
    z = np.load(cache)
    s = slice(None, limit)
    return z["q"][s], z["a"][s], z["r"][s]


def augment(q, a, rng):
    """Random Sudoku symmetry: digit relabel, band/stack and row/col-within-band shuffles, transpose."""
    B = q.shape[0]
    perm = np.stack([np.concatenate([[0], rng.permutation(9) + 1]) for _ in range(B)])
    def line_perm():
        bands = rng.permutation(3)
        return np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    rows = np.stack([line_perm() for _ in range(B)]); cols = np.stack([line_perm() for _ in range(B)])
    idx = (rows[:, :, None] * 9 + cols[:, None, :]).reshape(B, 81)
    tr = rng.random(B) < 0.5
    idx[tr] = idx[tr].reshape(-1, 9, 9).transpose(0, 2, 1).reshape(-1, 81)
    bi = np.arange(B)[:, None]
    return perm[bi, q[bi, idx]], perm[bi, a[bi, idx]]


def is_solved(q, pred):
    """pred digits 1-9 [N,81]: True where every unit is a permutation and all givens are kept."""
    keeps = ((q == 0) | (q == pred)).all(1)
    units = np.sort(pred[:, UNITS], axis=2)
    return keeps & (units == np.arange(1, 10)).all((1, 2))


def violations(q, pred):
    """Number of broken constraints per puzzle (duplicate digits in a unit + overwritten givens)."""
    u = np.sort(pred[:, UNITS], axis=2)
    dup = (u[:, :, 1:] == u[:, :, :-1]).sum((1, 2))
    return dup + ((q != 0) & (q != pred)).sum(1)


def show(grid):
    g = np.asarray(grid).reshape(9, 9); out = []
    for r in range(9):
        if r and r % 3 == 0: out.append("------+-------+------")
        out.append(" ".join(("." if v == 0 else str(v)) + (" |" if c in (2, 5) else "") for c, v in enumerate(g[r])))
    return "\n".join(out)


# Constraint groups as a factor graph: (variable, group, group type) for every membership.
# Type 0 = row, 1 = column, 2 = box. Any all-different style problem can be described this way.
N_GROUP_TYPES = 3
MEMBERSHIP = np.array([(v, g, g // 9) for g, unit in enumerate(UNITS) for v in unit]).T   # 3 x 243

RATING_BANDS = [(0, 0, "no guessing"), (1, 10, "1-10 backtracks"), (11, 50, "11-50 backtracks"), (51, 10 ** 9, ">50 backtracks")]
