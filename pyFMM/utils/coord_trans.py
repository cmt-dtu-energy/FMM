import numpy as np


def cart_to_sphe(X, eps=1e-12):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Input:
        X : array of shape (N, 3) representing N points in Cartesian coordinates (x, y, z)
        eps : small threshold to avoid division by zero
    Output:
        res : array of shape (N, 3) representing N points in spherical coordinates (r, theta, phi)
              r is the radial distance
              theta is the polar angle
              phi is the azimuthal angle
    """
    #-------- view of input array ----------------
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]
    #--------------------------------------------
    #-------- allocate output array -------------
    res = np.empty_like(X)
    #--------------------------------------------
    #---------- compute spherical coordinates -------------
    res[:,0] = np.sqrt(x**2 + y**2 + z**2)     # radial distance
    res[:,1] = np.arccos( z / res[:,0] )       # polar angle
    res[:,2] = np.mod(np.arctan2( y, x ), 2*np.pi)              # azimuthal angle
    #res[:,2] = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))  # azimuthal angle
    #------------------------------------------------------
    #------- handle small r case to avoid NaNs ----------------
    mask = res[:,0] < eps
    res[mask,1] = 0.0  # define polar angle at origin to be 0
    res[mask,2] = 0.0  # define azimuthal angle at origin to be 0
    #----------------------------------------------------------
    return res


def sphe_to_cart(X, eps=1e-12):
    """
    Convert spherical coordinates to Cartesian coordinates.
    Input:
        X : array of shape (N, 3) representing N points in spherical coordinates (r, theta, phi)
            r      : radial distance
            theta  : polar angle (angle from the z-axis, [0, π])
            phi    : azimuthal angle (angle from the x-axis in the x-y plane, [−π, π])
        eps : small threshold to avoid numerical issues for very small r
    Output:
        res : array of shape (N, 3) representing N points in Cartesian coordinates (x, y, z)
    """
    #-------- view of input array ----------------
    r = X[..., 0]
    theta = X[..., 1]
    phi = X[..., 2]
    #---------------------------------------------
    #-------- allocate output array --------------
    res = np.empty_like(X)
    #---------------------------------------------
    #-------- compute Cartesian coordinates -------
    sin_theta = np.sin(theta)
    res[:, 0] = r * sin_theta * np.cos(phi)  # x
    res[:, 1] = r * sin_theta * np.sin(phi)  # y
    res[:, 2] = r * np.cos(theta)            # z
    #---------------------------------------------
    #-------- handle small r case ----------------
    mask = r < eps
    res[mask] = 0.0
    #---------------------------------------------
    return res