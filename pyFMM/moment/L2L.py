import numpy as np

# Prefer package-relative utils when used inside pyFMM; fall back to
# top-level imports to allow running the file directly for quick tests.
try:
    # import the utils module so we don't pollute this module's namespace
    from .. import utils as utils
except Exception:
    import utils as utils  # type: ignore


def L2L_sphe(L_x0, x0, x1):
    """
    Translate local expansion L_x0 about x0 to expansion about x1.
    Assumes L is based on spherical harmonics.
    input:
        L_x0 : ((p+1)^2,) ndarray
            Local expansion moments about x0, flattened with index n^2 + n + m order.
        x0 : (3,) ndarray
            Cartesian coordinates of the original expansion centre.
        x1 : (3,) ndarray
            Cartesian coordinates of the new expansion centre.
    output:
        L_x1 : ((p+1)^2,) ndarray
            Translated local expansion moments about x1, flattened with index n^2 + n + m order.
    """
    #---------- determine order p from L -------------
    p = int(np.sqrt(L_x0.shape[0])) - 1
    #-------------------------------------------------
    #---------- get translation vector in spherical coords -------------
    x_diff = x0 - x1
    rho, alpha, beta = utils.cart_to_sphe(x_diff.reshape(1,3)).flatten()
    #-------------------------------------------------------------------
    #---------- allocate output array -------------
    L_x1 = np.zeros_like(L_x0)
    #----------------------------------------------
    #--------------- outer loops over output moments ----------------
    for j in range(p+1):
        for k in range(-j, j+1):
            acc = 0.0 + 0.0j
            #--------------- inner loops over input moments ----------------
            for n in range(j, p+1):
                jn = n - j
                for m in range(-n, n+1):
                    mk = m - k
                    if abs(mk) > jn:
                        continue
                    O = L_x0[utils.lm_index(n, m)]
                    phase = 1j**( abs(m) - abs(mk) - abs(k) )
                    coeff = utils.A_nm(jn,mk) * utils.A_nm(j,k) / utils.A_nm(n,m)  * (rho**jn)
                    one_coeff = (-1)**(n+j)
                    Y = utils.sph_harm_dir(jn, mk, alpha, beta)
                    acc += O * coeff * one_coeff * phase * Y
            #---------------------------------------------------------------
            L_x1[utils.lm_index(j, k)] = acc
    return L_x1