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




def H_dip(n,k, X, d, q):
    """
    Computes the 'H-contribution for the dipole to the magnetic moment
    inputs:
        n, k: degree and order of the spherical harmonic.
        X: array of points in spherical coordinates (rho, alpha, beta).
        d: dipole moment vector. Should be normalized
        q: scalar strength of the dipole
    outputs:
        res: sum of H-contributions at center point.
    """

    N = X.shape[0]

    #TODO - maybe add some functionality to handle n = 1 case - e.i. where X is just a single point
    H = np.zeros((N, 3), dtype=np.complex128)


    #------------- view of input array ----------------
    rho = X[..., 0]; alpha = X[..., 1]; beta = X[..., 2] 
    #--------------------------------------------------


    H[:,0] = np.cos(alpha) * np.cos(beta) * utils.dY_dtheta(n, k, alpha, beta) - (np.sin(beta)/np.sin(alpha)) * utils.dY_dphi(n, k, alpha, beta)
    H[:,1] = np.cos(alpha) * np.sin(beta) * utils.dY_dtheta(n, k, alpha, beta) + (np.cos(beta)/np.sin(alpha)) * utils.dY_dphi(n, k, alpha, beta)
    H[:,2] = - np.sin(alpha) * utils.dY_dtheta(n, k, alpha, beta)

    
    #TODO - technically missing a factor 2 here, but if this is absorbed in q it is ok
    res = q * np.sum(H * d, axis=1) * rho**(n-1)
    return np.sum(res)


def P2M_dip_sphe(X, d, q, p=4):
    """
    Compute multipole expansion from dipole sources given in spherical coords.
    Uses spherical harmonics to compute the multipole moments.
    input:
        X : (N, 3) ndarray
            Spherical coordinates of source points (r, theta, phi).
            where theta is the polar angle (0 <= theta <= pi)
            and   phi   is the azimuthal angle (0 <= phi < 2pi
        d : (N, 3) ndarray
            Dipole moment vectors at points X. Should be normalized.
            Is in the Cartesian basis.
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

    #---------- compute d dot p ---------------------------------------------
    dot_prod = q * r * ( d[:,0] * np.sin(theta) * np.cos(phi) + d[:,1] * np.sin(theta) * np.sin(phi) + d[:,2] * np.cos(theta) )
    #-----------------------------------------------------------------------


    #----------- allocate output array -------------
    M = np.zeros(((p+1)**2,), dtype=np.complex128)
    #----------------------------------------------
    for n in range(1,p+1): # start from n=1 since there is no monopole term for dipoles
        #--------- r^n term ----------------
        rpow = r**(n-2)
        #--------------------------------

        nmupi = n * dot_prod * rpow  # elementwise over particles


        #--------- compute moments ----------------
        for m in range(-n, n+1):
            Y = utils.sph_harm_dir(n, -m, theta, phi)
            M[utils.lm_index(n, m)] = np.sum(nmupi * Y) + H_dip(n,-m, X, d, q)
        #----------------------------------------

    return M