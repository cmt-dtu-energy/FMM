import numpy as np

# Prefer package-relative utils when used inside pyFMM; fall back to
# top-level imports to allow running the file directly for quick tests.
try:
    # import the utils module so we don't pollute this module's namespace
    from .. import utils as utils
except Exception:
    import utils as utils  # type: ignore

def M2M_sphe(M, x0, x1):
    """
    Translate multipole moments M about x0 to moments about x1.
    Assumes M is based on spherical harmonics.
    input:
        M : ((p+1)^2,) ndarray
            Multipole moments about x0, flattened with index n^2 + n + m order.
        x0 : (3,) ndarray
            Cartesian coordinates of the original expansion centre.
        x1 : (3,) ndarray
            Cartesian coordinates of the new expansion centre.
    output:
        M_trans : ((p+1)^2,) ndarray
            Translated multipole moments about x1, flattened with index n^2 + n + m order.
    """
    #---------- determine order p from M -------------
    p = int(np.sqrt(M.shape[0])) - 1
    #-------------------------------------------------
    #---------- get translation vector in spherical coords -------------
    x_diff = x0 - x1
    rho, alpha, beta = cart_to_sphe(x_diff.reshape(1,3)).flatten()
    #-------------------------------------------------------------------
    #---------- allocate output array -------------
    M_trans = np.zeros_like(M)
    #----------------------------------------------
    #---------- outer loops over output moments -------------
    for j in range(p+1):
        for k in range(-j, j+1):
            acc = 0.0 + 0.0j
            #---------- inner loops over input moments -------------
            for n in range(0, j+1):
                jn = j - n
                for m in range(-n, n+1):
                    km = k - m
                    if abs(km) > jn:
                        continue
                    O = M[lm_index(jn, km)]
                    phase = 1j**(abs(k) - abs(m) - abs(km))
                    coeff = (phase * A_nm(jn, km) * A_nm(n, m) / A_nm(j, k)) * (rho**n)
                    acc += O * coeff * sph_harm_dir(n, -m, alpha, beta)
            #----------------------------------------------
            M_trans[lm_index(j, k)] = acc 
        #----------------------------------------------
    #----------------------------------------------------------
    return M_trans