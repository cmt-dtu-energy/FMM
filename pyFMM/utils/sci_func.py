
from scipy.special import lpmv
from math import factorial as fac
import numpy as np


def sph_harm_dir(n, m, theta, phi):
    Pnm =  lpmv(abs(m), n, np.cos(theta))
    sqrt_fac = np.sqrt( fac(n - abs(m)) / fac(n + abs(m)) )
    return sqrt_fac * Pnm * np.exp(1j * m * phi)