
import numpy as np
from .. import utils as utils


def P_L_moment(L, x_tgt):
    """
    Evaluate potential at targets x_tgt from local expansion moments L.
    inputs:
        L: local expansion moments flattened with index n^2 + n + m order.
        x_tgt: (Ntgt, 3) array of target positions in spherical coords (r, theta, phi).
                spherical coord are assumed to be relative to the center of the local expansion.
    outputs:
        P: (Ntgt,) array of potentials at target positions. 
    """ 
    #------------ view of input array ----------------
    r = x_tgt[...,0]; theta = x_tgt[...,1]; phi = x_tgt[...,2]
    #-------------------------------------------------
    #----------- allocate output array -------------
    P = np.zeros(x_tgt.shape[0], dtype=np.complex128)
    #----------------------------------------------
    #---------- determine order p from L -------------
    p = int(np.sqrt(L.shape[0])) - 1
    #-------------------------------------------------
    #---------- compute potential -------------
    for n in range(p+1):
        #--------- r^(n) term ----------------
        r_pow = r**(n)
        #---------------------------------------
        for m in range(-n, n+1):
            Y = utils.sph_harm_dir(n, m, theta, phi)
            P += L[utils.lm_index(n, m)] * r_pow * Y
    return np.real(P)
    #----------------------------------------------
