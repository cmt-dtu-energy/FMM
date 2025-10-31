import numpy as np

# Prefer package-relative utils when used inside pyFMM; fall back to
# top-level imports to allow running the file directly for quick tests.
try:
    # import the utils module so we don't pollute this module's namespace
    from .. import utils as utils
except Exception:
    import utils as utils  # type: ignore


def M2L_sphe(M, x0, x1):
    """
    Evalua te local expansion L at x1 from multipole expansion M at x0.
    Assumes both M and L are based on spherical harmonics.
    input:
        M : ((p+1)^2,) ndarray
            Multipole moments about x0, flattened with index n^2 + n + m order.
        x0 : (3,) ndarray
            Cartesian coordinates of the multipole expansion centre.
        x1 : (3,) ndarray
            Cartesian coordinates of the local expansion centre.
    output:
        L : ((p+1)^2,) ndarray
            Local expansion moments about x1, flattened with index n^2 + n + m order
    """
    #---------- determine order p from M -------------
    p = int(np.sqrt(M.shape[0])) - 1
    #-------------------------------------------------
    #---------- get translation vector in spherical coords -------------
    x_diff = x0 - x1 
    rho, alpha, beta = utils.cart_to_sphe(x_diff.reshape(1,3)).flatten()
    #-------------------------------------------------------------------
    #---------- allocate output array -------------
    L = np.zeros_like(M)
    #----------------------------------------------
    #----------- outer loops over output local moments -------------
    for j in range(p+1):
        for k in range(-j, j+1):
            acc = 0.0 + 0.0j
            A_kj = utils.A_nm(j, k)
            #----------- inner loops over input multipole moments -------------
            for n in range(0, p+1):
                jn = j + n
                #--------- r^( -(jn + 1) ) term ----------------
                rho_pow = 1.0 / ( rho**(jn + 1) )
                #-----------------------------------------------

                one_coeff = (-1)**n

                for m in range(-n, n+1):
                    mk = m - k
                    O = M[utils.lm_index(n, m)]
                    phase = 1j**( abs(mk) - abs(k) - abs(m) )
                    coeff = utils.A_nm(n,m) * A_kj / ( utils.A_nm(jn, mk) )
                    Y = utils.sph_harm_dir(jn, mk, alpha, beta)
                    acc += O * coeff * one_coeff * phase * rho_pow * Y
            #---------------------------------------------------------------
            L[utils.lm_index(j, k)] = acc
    #----------------------------------------------------------
    return L