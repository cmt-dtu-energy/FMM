import numpy as np

# Import coord_trans as a module and re-export its functions from this module.
# We also set the __module__ attribute on the exported functions so that
# introspection (e.g., show_package_tree) reports them as belonging to
# `pyFMM.utils.point_sampling` rather than `pyFMM.utils.coord_trans`.
from . import coord_trans as coord_trans

# re-export functions and adjust their __module__ so they appear as defined
# in this module for introspection tools
cart_to_sphe = coord_trans.cart_to_sphe
sphe_to_cart = coord_trans.sphe_to_cart
try:
    cart_to_sphe.__module__ = __name__
    sphe_to_cart.__module__ = __name__
except Exception:
    # Some builtins or wrapped callables may not allow assignment; ignore in that case
    pass


def generate_points_in_sphere(n_points, radius, r_min=0.0, center=None, spherical=False, eps=1e-12):
    """
    Generate random points uniformly distributed inside a sphere (or shell).

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    radius : float
        Outer radius of the sphere (> 0).
    r_min : float, optional
        Inner radius (>= 0). Use r_min > 0 for a hollow shell. Default: 0.0
    center : array-like of shape (3,), optional
        Cartesian coordinates of the sphere centre. If None, assume origin.
        Only used when returning Cartesian coordinates.
    spherical : bool, optional
        If True, return spherical coordinates (r, theta, phi) about the sphere's centre.
        If False, return Cartesian (x, y, z).
    eps : float, optional
        Small guard for numerical stability.

    Returns
    -------
    points : (n_points, 3) ndarray
        If spherical=False: Cartesian points (x, y, z).
        If spherical=True : Spherical points (r, theta, phi) relative to the sphere centre.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if r_min < 0 or r_min > radius:
        raise ValueError("r_min must satisfy 0 <= r_min <= radius")

    r = np.cbrt(np.random.uniform( low=r_min**3, high=radius**3, size=n_points))
    theta = np.arccos( np.random.uniform(-1,1,n_points) )
    phi   = np.random.uniform(0,2*np.pi,n_points)

    #------------ if spherical and center is None, return directly -------------
    if spherical and center is None:
        pts = np.column_stack((r, theta, phi))
    #---------------------------------------------------------------------------
    #------------ otherwise convert to cartesian --------------------
    else:
        pts_cart = sphe_to_cart( np.column_stack((r, theta, phi)), eps=eps )
        if spherical:
            pts_cart += np.asarray(center).reshape(1,3)
            pts = cart_to_sphe(pts_cart, eps=eps)
        else:
            if center is not None:
                pts_cart += np.asarray(center).reshape(1,3)
            pts = pts_cart
    return pts