import numpy as np

def lm_index(n, m):
    return n*n + n + m



def dipoles_to_monopole_pairs(X, d, q, delta_mon = 1e-6, mode="match_moment"):
    """
    Convert dipoles at positions X with directions d and scalars q
    into two monopoles at X ± delta_mon * d_hat, with charges ±Q.
    Parameters
    ----------
    X : (N, 3) array
        Dipole reference positions (centres).
    d : (N, 3) array
        Dipole direction vectors (need not be unit; will be normalised).
    q : (N,) array
        Dipole scalars. Interpretation depends on `mode`.
    delta_mon : float
        Half-separation of the monopole pair (in same units as X).
    mode : {"match_moment", "reuse_q"}
        - "match_moment": Q = q / (2*delta_mon)  -> dipole moment = q * d_hat
        - "reuse_q":      Q = q                  -> dipole moment = (2*delta_mon*q) * d_hat

    Returns
    -------
    X_mono : (2N, 3) array
        Monopole positions: first all + charges, then all - charges.
    q_mono : (2N,) array
        Monopole charges: [+Q_0, ..., +Q_{N-1}, -Q_0, ..., -Q_{N-1}]
    """
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float)
    q = np.asarray(q, dtype=float)

    # normalise directions safely
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    # avoid division by zero: any zero-length d gets replaced by a default direction
    safe = d_norm.squeeze() > 0
    d_hat = np.zeros_like(d)
    d_hat[safe] = d[safe] / d_norm[safe]
    if not np.all(safe):
        # default any zero vector to x-hat (or pick random unit vectors if preferred)
        d_hat[~safe] = np.array([1.0, 0.0, 0.0])

    if mode == "match_moment":
        Q = q / (2.0 * float(delta_mon))
    elif mode == "reuse_q":
        Q = q
    else:
        raise ValueError("mode must be 'match_moment' or 'reuse_q'")

    # positions for +Q and -Q
    X_plus  = X + delta_mon * d_hat
    X_minus = X - delta_mon * d_hat

    # stack: first all +, then all -
    X_mono = np.vstack([X_plus, X_minus])
    q_mono = np.concatenate([+Q, -Q])

    return X_mono, q_mono
