
from scipy.special import lpmv
from math import factorial as fac
import numpy as np
from math import lgamma  # log-gamma; lgamma(k+1) = log(k!)


def sph_harm_dir(n, m, theta, phi):
    Pnm =  lpmv(abs(m), n, np.cos(theta))
    sqrt_fac = np.sqrt( fac(n - abs(m)) / fac(n + abs(m)) )
    return sqrt_fac * Pnm * np.exp(1j * m * phi)


def dx_P(n, m, x):
    if n == 0:
        return 0.0
    else:
        #TODO - need some  check if x**2-1 is zero
        return 1.0 / (x**2 - 1.0) * ( - ( n+ 1) * x * lpmv(m, n, x) + (n - m + 1) * lpmv(m, n + 1, x) )




def dY_dtheta(n, m, theta, phi):
    """
    Compute the theta-derivative of the spherical harmonic Y_n^m(theta, phi).
    inputs:
        n, m: degree and order of the spherical harmonic.
        theta, phi: arrays of polar and azimuthal angles.
    outputs:
        dY_dtheta: array of the theta-derivative of Y_n^m at the input angles.
    """

    sqrt_fac = np.sqrt( fac(n - abs(m)) / fac(n + abs(m)) )
    dP = dx_P(n, abs(m), np.cos(theta))
    res = - sqrt_fac * dP * np.sin(theta) * np.exp(1j * m * phi)
    return res

def dY_dphi(n, m, theta, phi):
    """
    Compute the phi-derivative of the spherical harmonic Y_n^m(theta, phi).
    inputs:
        n, m: degree and order of the spherical harmonic.
        theta, phi: arrays of polar and azimuthal angles.
    outputs:
        dY_dphi: array of the phi-derivative of Y_n^m at the input angles.
    """
    Y = sph_harm_dir(n, m, theta, phi)
    return 1j * m * Y



def A_nm(n, m):
    """
    A_n^m = (-1)^n / sqrt((n-m)! (n+m)!)
    computed stably via log-gamma to avoid overflow/ufunc issues.
    """
    if abs(m) > n:
        return 0.0  # outside domain; safe guard
    # log of the denominator factorial product
    log_den = lgamma(n - m + 1) + lgamma(n + m + 1)
    return ((-1)**n) * np.exp(-0.5 * log_den)