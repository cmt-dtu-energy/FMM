
from scipy.special import lpmv
from math import factorial as fac
import numpy as np
from math import lgamma  # log-gamma; lgamma(k+1) = log(k!)


def sph_harm_dir(n, m, theta, phi):
    Pnm =  lpmv(abs(m), n, np.cos(theta))
    sqrt_fac = np.sqrt( fac(n - abs(m)) / fac(n + abs(m)) )
    return sqrt_fac * Pnm * np.exp(1j * m * phi)


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