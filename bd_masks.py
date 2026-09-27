"""Block-diffusion attention masks and token-shift indexing (NumPy, no torch dependency).

Training uses the vectorised two-copy layout [x_noisy (L) | x_clean (L)]:
  * noisy query in block b  -> sees noisy keys in block b (bidirectional) + clean keys in blocks < b
  * clean query at i        -> sees clean keys at j <= i (plain causal: the AR ability is preserved)
Token shift: the masked token at position j is predicted from the logits at position j-1, so the
converted model keeps the next-token head it was pretrained with.
  * j is the first token of its block -> logits come from the CLEAN copy at j-1 (a pure AR prediction)
  * otherwise                        -> logits come from the NOISY copy at j-1 (same block, bidirectional)
"""
import numpy as np


def block_ids(L, B):
    return np.arange(L) // B


def train_mask(L, B):
    """Boolean (2L, 2L) mask, True = may attend."""
    b = block_ids(L, B)
    m = np.zeros((2 * L, 2 * L), dtype=bool)
    m[:L, :L] = b[:, None] == b[None, :]                 # noisy -> noisy, same block
    m[:L, L:] = b[None, :] < b[:, None]                  # noisy -> clean, earlier blocks
    m[L:, L:] = np.tril(np.ones((L, L), dtype=bool))     # clean -> clean, causal
    return m


def logit_index(j, L, B):
    """Row (in the 2L layout) whose logits predict token j of the noisy copy."""
    if j <= 0:
        raise ValueError("position 0 has no predecessor")
    return L + j - 1 if j % B == 0 else j - 1


def logit_indices(positions, L, B):
    return np.array([logit_index(int(j), L, B) for j in positions], dtype=np.int64)


def infer_mask(n_ctx, B):
    """Boolean (n_ctx+B, n_ctx+B) mask for sampling one block after a clean context.
    Context is causal among itself; the block sees all context and itself bidirectionally."""
    n = n_ctx + B
    m = np.zeros((n, n), dtype=bool)
    m[:n_ctx, :n_ctx] = np.tril(np.ones((n_ctx, n_ctx), dtype=bool))
    m[n_ctx:, :] = True
    return m


def infer_logit_index(k, n_ctx):
    """Row whose logits predict block position k (token shift)."""
    return n_ctx + k - 1


def sample_block_noise(resp_mask, B, rng, t_min=0.05):
    """Per-block masking ratio t ~ U(t_min, 1); only response tokens are ever masked.
    Guarantees at least one masked token per example that has a response."""
    L = resp_mask.shape[0]
    noisy = np.zeros(L, dtype=bool)
    for s in range(0, L, B):
        blk = resp_mask[s:s + B]
        if not blk.any():
            continue
        t = rng.uniform(t_min, 1.0)
        noisy[s:s + B] = blk & (rng.random(blk.shape[0]) < t)
    if resp_mask.any() and not noisy.any():
        idx = np.flatnonzero(resp_mask)
        noisy[rng.choice(idx)] = True
    noisy[0] = False
    return noisy


def additive(mask_bool, neg=-1e4):
    """Convert a boolean mask to an additive float mask (0 = attend, neg = blocked)."""
    return np.where(mask_bool, 0.0, neg).astype(np.float32)
