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


# ----------------- Elementary rotations (kept separate) -----------------
def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=float)

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

# ------------- Internal: rotation about arbitrary axis (Rodrigues) -------------
def R_axis(axis, angle):
    """
    Rotate by 'angle' around 3D 'axis' (arbitrary direction).
    """
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm == 0:
        return np.eye(3)
    a = axis / nrm
    x, y, z = a
    K = np.array([[ 0, -z,  y],
                  [ z,  0, -x],
                  [-y,  x,  0]], dtype=float)
    c = np.cos(angle); s = np.sin(angle)
    return np.eye(3) + s * K + (1 - c) * (K @ K)

# ----------------- 1) Make a 2D grid on a standard axis plane ------------------
def make_plane_grid(LLC, URC, N_grid, plane="xy"):
    """
    Create a square N_grid x N_grid grid in one of the axis-aligned planes
    with half-cell centering identical to your original code.

    LLC, URC : (3,) arrays defining the bounding box (as in your code).
    N_grid   : int, number of cells per side.
    plane    : 'xy', 'xz', or 'yz'.

    Returns
    -------
    grid_points : (N_grid*N_grid, 3) array of 3D points.
    X_grid, Y_grid : (N_grid, N_grid) 2D arrays of the two in-plane coords.
    """
    LLC = np.asarray(LLC, dtype=float)
    URC = np.asarray(URC, dtype=float)
    size = URC - LLC
    dx, dy, dz = size / N_grid

    if plane.lower() == "xy":
        x = np.linspace(LLC[0] + dx/2, URC[0] - dx/2, N_grid)
        y = np.linspace(LLC[1] + dy/2, URC[1] - dy/2, N_grid)
        Xg, Yg = np.meshgrid(x, y, indexing='xy')
        gp = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros_like(Xg).ravel()])

    elif plane.lower() == "xz":
        x = np.linspace(LLC[0] + dx/2, URC[0] - dx/2, N_grid)
        z = np.linspace(LLC[2] + dz/2, URC[2] - dz/2, N_grid)
        Xg, Zg = np.meshgrid(x, z, indexing='xy')
        gp = np.column_stack([Xg.ravel(), np.zeros_like(Xg).ravel(), Zg.ravel()])
        # For return consistency, we name them X_grid, Y_grid (but here Y_grid==Z)
        Yg = Zg

    elif plane.lower() == "yz":
        y = np.linspace(LLC[1] + dy/2, URC[1] - dy/2, N_grid)
        z = np.linspace(LLC[2] + dz/2, URC[2] - dz/2, N_grid)
        Yg, Zg = np.meshgrid(y, z, indexing='xy')
        gp = np.column_stack([np.zeros_like(Yg).ravel(), Yg.ravel(), Zg.ravel()])
        # Return as X_grid, Y_grid (but here X_grid==Y, Y_grid==Z)
        Xg = Yg; Yg = Zg

    else:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

    return gp, Xg, Yg

# ------------- 2) Rotate a plane by (theta, phi) with polar-tilt model ---------
def rotate_plane(theta, phi, center=None):
    """
    Build the rotation that:
      - first spins by 'phi' about the global z-axis,
      - then tilts by 'theta' about an axis lying in the XY plane,
        specifically the 'spun x-axis' (i.e., axis = Rz(phi) @ e_x).

    This treats theta as a 'polar tilt' away from the XY plane, and phi as the
    rotation around z.

    Returns
    -------
    R : (3,3) rotation matrix (active; p' = R @ p)
    u, v, n : plane basis vectors in world coords
              u = first in-plane axis (spun x, then tilted)
              v = second in-plane axis
              n = plane normal
    center : (3,) plane origin (defaults to 0 vector)
    """
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)

    # spin axis and in-plane basis before tilt
    ex = np.array([1., 0., 0.])
    ey = np.array([0., 1., 0.])
    ez = np.array([0., 0., 1.])

    Rz_phi = Rz(phi)
    u_spin = Rz_phi @ ex    # axis in XY plane we tilt about
    v_spin = Rz_phi @ ey    # companion in-plane axis before tilt
    n_spin = ez             # normal before tilt

    # tilt around u_spin by theta
    Ru = R_axis(u_spin, theta)

    # full rotation that maps original basis into final plane
    R = Ru @ Rz_phi

    # final basis
    u = R @ ex
    v = R @ ey
    n = R @ ez

    return R, u, v, n, center

# ----------------- 3) Project 3D points onto a plane (u, v, center) ------------
def project_points_onto_plane(X, u, v, center=None):
    """
    Project 3D points onto the 2D coordinates of a plane spanned by (u, v).

    Parameters
    ----------
    X : (N,3) array
    u, v : (3,) arrays, orthonormal in-plane basis vectors
    center : (3,) array, plane origin. If None -> (0,0,0).

    Returns
    -------
    X_2D : (N,2) array of coordinates (xi, eta) in the (u,v) basis.
    """
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    X_rel = np.asarray(X, dtype=float) - center
    xi  = X_rel @ np.asarray(u, dtype=float)
    eta = X_rel @ np.asarray(v, dtype=float)
    return np.column_stack([xi, eta])
