"""Unit tests for spherical-harmonic multipole operations (P2M, M2M, M2L, L2L).

These tests use a small random source cluster and a few target points. They
compare potentials computed via multipole/local expansions against the direct
pairwise sum. The allowed error is set relative to the theoretical multipole
truncation scaling ~ (a / R)^(p+1) where 'a' is the source cluster radius and
R is the distance from the expansion centre to the evaluation points.
"""
import numpy as np
import sys
import os
import pytest
import importlib
from pathlib import Path


def _add_repo_root_to_syspath(marker_dir_name: str = "pyFMM"):
    """Search upward from this file for a directory containing `marker_dir_name`
    and insert that parent into sys.path so tests can import the package.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / marker_dir_name).is_dir():
            repo_root = str(p)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return repo_root
    return None


_add_repo_root_to_syspath()


def _import_pyfmm_modules():
    """Attempt to import pyFMM and required symbols. Fail the test run with a
    clear message if import fails so the user can see the underlying exception.
    """
    try:
        # Import the package and required public symbols
        mod = importlib.import_module("pyFMM")
        from pyFMM.moment import P2M_sphe, M2M_sphe, M2L_sphe, L2L_sphe
        from pyFMM.pot_eval import P_direct_cart, P_sphe, P_L_sphe
        from pyFMM.utils import cart_to_sphe, sphe_to_cart, sph_harm_dir, lm_index
        return P2M_sphe, M2M_sphe, M2L_sphe, L2L_sphe, P_direct_cart, P_sphe, P_L_sphe, cart_to_sphe, sphe_to_cart, sph_harm_dir, lm_index
    except Exception as e:
        # Raise an explicit pytest failure so test output contains the import error
        pytest.fail(f"Failed to import pyFMM or its submodules: {e}")


# perform imports and bind names for the tests
P2M_sphe, M2M_sphe, M2L_sphe, L2L_sphe, P_direct_cart, P_sphe, P_L_sphe, cart_to_sphe, sphe_to_cart, sph_harm_dir, lm_index = _import_pyfmm_modules()


N = 10_000
source_area_size =  1.0
LLC = np.array([ -1.0, -1.0, -1.0 ]) * source_area_size   # Lower Left Corner
URC = np.array([  1.0,  1.0,  1.0 ]) * source_area_size   # Upper Right Corner


N_tgt = 20
r_min = 3.0 * source_area_size *  np.sqrt(3) 
r_max = 15.0 * source_area_size *  np.sqrt(3)

p_min = 1
p_max = 10



def make_sources(N, LLC, URC):
    X = np.random.uniform(low=LLC, high=URC, size=(N, 3))
    q = np.random.choice([-1, 1], size=N)
    X_sphe = cart_to_sphe(X)
    return X, X_sphe, q

X, x_polar, q = make_sources(N, LLC, URC)
A = np.sum(np.abs(q))


def make_local_expansion_targets(N_tgt, r_min, r_max):
    r = np.linspace(r_min, r_max, N_tgt)
    theta = np.pi * np.random.uniform(size=N_tgt)
    phi = 2.0 * np.pi * np.random.uniform(size=N_tgt)
    L_sphe = np.vstack((r, theta, phi)).T
    L_cart = sphe_to_cart(L_sphe)
    return L_cart, L_sphe


def make_child_points(L_cart, max_offset):
    N_tgt = L_cart.shape[0]
    offsets = np.random.uniform(low=-max_offset, high=max_offset, size=(N_tgt, 3))
    eval_points = L_cart + offsets
    return eval_points


L_cart, L_sphe = make_local_expansion_targets(N_tgt=N_tgt, r_min=r_min, r_max=r_max)
child_points = make_child_points(L_cart, max_offset=0.5 * source_area_size)
eval_points = make_child_points(L_cart, max_offset=0.5 * source_area_size)
eval_points_polar = cart_to_sphe(eval_points)


def P2M_err_theory(A, r, a, p):
    """Estimate of the relative error in potentials computed via multipole
    expansion of order p for a source cluster of radius a evaluated at distance
    r from the expansion centre.
    """
    return ( A / (r - a)) * ( a / r)**(p+1)


def M2M_err_theory(A, r, rho, a, p):
    """Estimate of the relative error in potentials computed via multipole
    expansion of order p after a M2M translation over distance rho for a source
    cluster of radius a evaluated at distance r from the new expansion centre.
    """
    return ( A / ( r - ( a + rho))) * ( (a + rho) / r )**(p+1)



def M2L_err_theory(A, c, a, p):
    """Estimate of the relative error in potentials computed via local expansion
    of order p obtained via M2L translation over distance c for a source cluster
    of radius a.
    """
    return ( A / ( c*a-a)) * ( 1 / c)**(p+1)


def L2L_err_theory(A, c, a, p):
   return M2L_err_theory(A, c, a, p)



#------------------- test of P2M -------------------------------------------------------

def test_P2M():
    for p in range(p_min, p_max+1):
        M = P2M_sphe(x_polar, q, p=p)
        P_exp_sphe = P_sphe(M, eval_points_polar)
        P_direct = P_direct_cart(X, q, eval_points)


        # per-target absolute errors
        err = np.abs(P_exp_sphe - P_direct)

        # per-target distances from the expansion centre (origin)
        r_per_target = np.linalg.norm(eval_points - np.zeros((1, 3)), axis=1)
        a = source_area_size * np.sqrt(3)  # approx radius of source cluster
        err_theory = P2M_err_theory(A, r_per_target, a, p)
        assert np.all(err < np.maximum(1e-8, err_theory)), (
            f"P2M p={p}: some targets exceed theoretical bound; max(err/bound)={np.max(err/(err_theory+1e-16)):.3e}"
        )
#----------------------------------------------------------------------------------------


#------------------- test of M2M -------------------------------------------------------


#------------- first define child centres and masks -------------
child_centres = np.array([[-0.5, -0.5, -0.5],
                 [ 0.5, -0.5, -0.5],
                 [-0.5,  0.5, -0.5],
                 [ 0.5,  0.5, -0.5],
                 [-0.5, -0.5,  0.5],
                 [ 0.5, -0.5,  0.5],
                 [-0.5,  0.5,  0.5],
                 [ 0.5,  0.5,  0.5]]) * source_area_size
child_masks = []
for cx in child_centres:
    x_mask = np.logical_and(X[:,0] >= cx[0] - source_area_size *0.5, X[:,0] <= cx[0] + source_area_size * 0.5)   
    y_mask = np.logical_and(X[:,1] >= cx[1] - source_area_size *0.5, X[:,1] <= cx[1] + source_area_size * 0.5)
    z_mask = np.logical_and(X[:,2] >= cx[2] - source_area_size *0.5, X[:,2] <= cx[2] + source_area_size * 0.5)
    mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
    child_masks.append(mask)
#------------------------------------------------------------
#------------- helper functions ----------------
def get_child_moments(child_masks, p_in):
    child_moments = []
    for i, mask in enumerate(child_masks):
        X_child = X[mask]
        X_child_rel = X_child - child_centres[i]
        q_child = q[mask]
        M_child = P2M_sphe(cart_to_sphe(X_child_rel), q_child, p=p_in)
        child_moments.append(M_child)
    child_moments = np.array(child_moments)
    return child_moments
def get_parent_moments(child_moments, x1):
    M_parent = np.zeros_like(child_moments[0])
    for i, M_child in enumerate(child_moments):
        x0 = child_centres[i]
        M_trans = M2M_sphe(M_child, x0=x0, x1=x1)
        M_parent += M_trans
    return M_parent
#------------------------------------------------
#-------------- test function ----------------
# We test here M2M by forming child moments and translating them to
# a parent centre, then comparing potentials against direct sum.
def test_M2M():
    x0 = np.zeros(3)
    rho = np.sqrt( (source_area_size*0.5)**2 * 3 )

    for p in range(p_min, p_max+1):
        child_moments = get_child_moments(child_masks, p_in=p)
        M_tran = get_parent_moments(child_moments, x1=x0)


        P_exp_sphe = P_sphe(M_tran, eval_points_polar)
        P_direct = P_direct_cart(X, q, eval_points) 

        # per-target absolute errors
        err = np.abs(P_exp_sphe - P_direct)

        r_per_target = np.linalg.norm(eval_points - x0, axis=1)
        a = source_area_size * np.sqrt(3)
        err_theory = M2M_err_theory(A, r_per_target, rho, a, p)

        assert np.all(err < np.maximum(1e-8, err_theory)), (
            f"M2M p={p}: some targets exceed theoretical bound; max(err/bound)={np.max(err/(err_theory+1e-16)):.3e}"
        )

#----------------------------------------------------------------------------------------

def test_M2L():
    x0 = np.zeros(3)
    # Precompute per-target data once to avoid repeated work inside loops
    # direct potentials for all evaluation points (single call)
    P_direct_all = P_direct_cart(X, q, eval_points, eps=1e-10)
    # relative spherical coordinates of evaluation points w.r.t. each local centre
    eval_rel_sph = cart_to_sphe(eval_points - L_cart)
    # distances rho and c factors per local centre (independent of p)
    rho_arr = np.linalg.norm(L_cart - x0.reshape(1, 3), axis=1)
    a = source_area_size * np.sqrt(3)
    c_arr = rho_arr / a - 1.0

    for p in range(p_min, p_max + 1):
        M = P2M_sphe(x_polar, q, p=p)

        for i, x1 in enumerate(L_cart):
            M_loc = M2L_sphe(M, x0=x0, x1=x1)

            # get precomputed relative spherical coordinate for this eval point
            tgt_rel_sph = eval_rel_sph[i : i + 1]

            P_loc = P_L_sphe(M_loc, tgt_rel_sph)[0]
            P_direct = P_direct_all[i]
            err = np.abs(P_loc - P_direct)

            err_theory = M2L_err_theory(A, c_arr[i], a, p)

            assert err < max(1e-8, err_theory), (
                f"M2L p={p}: target {i} exceeds theoretical bound; err/bound={err/(err_theory+1e-16):.3e}"
            )



#----------------------------------------------------------------------------------------



def test_L2L():
    """Compute local expansions via M2L at each local centre, translate each
    local expansion to its child point, then evaluate potentials from the
    translated local moments and compare to direct sum.
    """
    x0 = np.zeros(3)
    for p in range(p_min, p_max + 1):
        M = P2M_sphe(x_polar, q, p=p)

        for i, x1 in enumerate(L_cart):
            # local expansion at local centre from multipole
            L_x1 = M2L_sphe(M, x0=x0, x1=x1)

            # translate local expansion to child expansion centre
            child_center = child_points[i]
            L_child = L2L_sphe(L_x1, x1, child_center)

            # evaluate at the corresponding evaluation point relative to child centre
            eval_point = eval_points[i]
            eval_rel = cart_to_sphe(np.array([eval_point - child_center]))[0]

            P_loc = P_L_sphe(L_child, np.array([eval_rel]))[0]
            P_direct = P_direct_cart(X, q, np.array([eval_point]), eps=1e-10)[0]

            err = np.abs(P_loc - P_direct)

            # theoretical bound using distance from child centre to original sources
            d = np.linalg.norm(child_center - x0)
            a = source_area_size * np.sqrt(3)
            err_theory = L2L_err_theory(A, d / a - 1.0, a, p)

            assert err < max(1e-8, err_theory), (
                f"L2L p={p}: target {i} exceeds theoretical bound; err/bound={err/(err_theory+1e-16):.3e}"
            )