
import numpy as np
from .. import utils as utils


def P_sphe(M, x_tgt):
    """
    Evaluate potential at targets x_tgt from multipole moments M.
    Assumes M given in spherical harmonic basis.
    Assumes x_tgt is given in spherical coordinates relative to the center of the multipole expansion.
    inputs:
        M: multipole moments flattened with index n^2 + n + m order.
        x_tgt: (Ntgt, 3) array of target positions in spherical coords (r, theta, phi).
    outputs:
        P: (Ntgt,) array of potentials at target positions. 
    """
    #------------ view of input array ----------------
    r = x_tgt[...,0]; theta = x_tgt[...,1]; phi = x_tgt[...,2]
    #-------------------------------------------------
    #----------- allocate output array -------------
    P = np.zeros(x_tgt.shape[0], dtype=np.complex128)
    #----------------------------------------------
    #---------- determine order p from M -------------
    p = int(np.sqrt(M.shape[0])) - 1
    #-------------------------------------------------
    #---------- compute potential -------------
    for n in range(p+1):
        #--------- r^-(n+1) term ----------------
        rinv = r**(-(n+1))
        #---------------------------------------
        for m in range(-n, n+1):
            Y = utils.sph_harm_dir(n, m, theta, phi)
            P += M[utils.lm_index(n, m)] * rinv * Y
    #----------------------------------------------
    return np.real(P)