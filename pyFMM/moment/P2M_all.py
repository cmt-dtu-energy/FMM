import numpy as np

# Prefer package-relative utils when used inside pyFMM; fall back to
# top-level imports to allow running the file directly for quick tests.
try:
    # import the utils module so we don't pollute this module's namespace
    from .. import utils as utils
except Exception:
    import utils as utils  # type: ignore


def P2M_sphe(X, q, p=4):
    """
    Compute multipole expansion from point sources given in spherical coords.
    Uses spherical harmonics to compute the multipole moments.
    input:
        X : (N, 3) ndarray
            Spherical coordinates of source points (r, theta, phi).
            where theta is the polar angle (0 <= theta <= pi)
            and   phi   is the azimuthal angle (0 <= phi < 2pi
        q : (N,) ndarray
            Source strengths at points X.
        p : int
            Order of the multipole expansion.
    output:
        M : ((p+1)^2,) ndarray
            Multipole moments flattened with index n^2 + n + m order.
    """
    #------------- view of input array ----------------
    r = X[..., 0]; theta = X[..., 1]; phi = X[..., 2] 
    #--------------------------------------------------
    #----------- allocate output array -------------
    M = np.zeros(((p+1)**2,), dtype=np.complex128)
    #----------------------------------------------
    for n in range(p+1):
        #--------- r^n term ----------------
        rpow = r**n if n > 0 else np.ones_like(r)
        #--------------------------------
        #--------- compute moments ----------------
        for m in range(-n, n+1):
            Y = utils.sph_harm_dir(n, -m, theta, phi)
            M[utils.lm_index(n, m)] = np.sum(q * rpow * Y)
        #----------------------------------------
    return M