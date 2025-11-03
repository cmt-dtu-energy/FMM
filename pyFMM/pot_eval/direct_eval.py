
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


def P_dipole_direct_cart(X, d, q, x_tgt, eps=1e-10):
    """
    Direct dipole potential:
        Phi(x) = sum_i ( q_i * d_i · (x - X_i) ) / ||x - X_i||^3
    X     : (N,3) source positions (Cartesian)
    d     : (N,3) dipole directions (Cartesian)
    q     : (N,)  dipole strengths (so ell_i = q_i * d_i)
    x_tgt : (Ntgt,3) target positions (Cartesian)
    eps   : softening to avoid 1/0; uses R = sqrt(||diff||^2 + eps^2)
    Returns: (Ntgt,) potential at each target
    """
    # (Ntgt, N, 3)
    diff = x_tgt[:, None, :] - X[None, :, :]

    # distances and softened cubes
    R2 = np.einsum('tni,tni->tn', diff, diff)  # ||diff||^2
    if eps != 0.0:
        R2 = R2 + eps*eps
    R   = np.sqrt(R2)
    R3  = R2 * R  # = (||diff||^2) * ||diff|| = ||diff||^3 (softened)

    # l_i · (x - X_i) with l_i = q_i d_i
    # shape (Ntgt, N): dot over the 3-vector
    ldotr = np.einsum('ni,tni->tn', q[:, None]*d, diff)

    return np.sum(ldotr / R3, axis=1)
