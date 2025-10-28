
import numpy as np

def P_direct_cart(X, q, x_tgt, eps=1e-10):
    """
    Fully-vectorized direct sum. May allocate a (Ntgt, N) array.
    eps: small softening to avoid 1/0 if targets coincide with sources.
    """
    # (Ntgt, 1, 3) - (1, N, 3) -> (Ntgt, N, 3)
    diff = x_tgt[:, None, :] - X[None, :, :]
    r = np.linalg.norm(diff, axis=2)  # (Ntgt, N)
    if eps != 0.0:
        r = np.sqrt(r*r + eps*eps)
    return (q / r).sum(axis=1)