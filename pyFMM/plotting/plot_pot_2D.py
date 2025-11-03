"""Plotting helpers for 2D potential heatmaps.

Provides a single function `plot_potential_2D(points, potential, axis, ...)`
which draws a heatmap of scattered potential values onto a provided
matplotlib `Axes` using a triangular mesh (no SciPy dependency).
"""
from typing import Optional, Sequence, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
except Exception:  # pragma: no cover - plotting optional in headless env
    plt = None  # type: ignore
    Triangulation = None  # type: ignore


def plot_potential_2D(points: Sequence[Sequence[float]], potential: Sequence[float], axis, *, cmap: str = "viridis", shading: str = "gouraud", scatter_points: bool = False, s: float = 8.0, vmin: Optional[float] = None, vmax: Optional[float] = None, **kwargs):
    """Plot a heatmap of scattered potential values on the given Axes.

    Parameters
    - points: sequence-like of shape (N,2) or (N,3). If 3D is given and one
      coordinate is (near-)constant it will be dropped automatically.
    - potential: sequence of length N with scalar values to plot.
    - axis: matplotlib Axes to draw on (required).
    - cmap: matplotlib colormap name.
    - shading: shading mode passed to `tripcolor` (e.g., 'gouraud' or 'flat').
    - scatter_points: if True, overlay the sampled points as scatter markers.
    - s: marker size for scatter overlay.
    - vmin, vmax: color scale limits (optional).
    - kwargs: additional keyword args forwarded to `tripcolor` (edgecolors, linewidth, etc.).

    Returns the mappable (QuadMesh or PolyCollection) added to the axis.
    """
    if axis is None:
        raise ValueError("An axis (matplotlib.axes.Axes) must be provided")
    if plt is None or Triangulation is None:
        raise RuntimeError("matplotlib is required for plotting (install matplotlib)")

    pts = np.asarray(points, dtype=float)
    pot = np.asarray(potential, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != pot.shape[0]:
        raise ValueError("points must be an (N,2) or (N,3) array and potential length must match N")

    # If 3D points are provided, detect collapsed dimension(s) and drop them
    if pts.shape[1] == 3:
        ranges = pts.ptp(axis=0)
        active = np.where(ranges > 1e-12)[0]
        if active.size >= 2:
            dims = active[:2]
            xy = pts[:, dims]
        elif active.size == 1:
            raise ValueError("Points appear 1D (only one varying coordinate) — use a 1D plotting routine")
        else:
            raise ValueError("Points are degenerate (all coordinates constant)")
    else:
        xy = pts

    x = xy[:, 0]
    y = xy[:, 1]

    # Triangulate the scattered points and use tripcolor to produce a smooth heatmap
    tri = Triangulation(x, y)
    # Use tripcolor which accepts triangulation and per-vertex values
    m = axis.tripcolor(tri, pot, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if scatter_points:
        axis.scatter(x, y, c='k', s=s, lw=0, alpha=0.6)

    axis.set_aspect('equal')
    # set reasonable limits (pad a bit)
    pad_x = (x.max() - x.min()) * 0.02 if x.max() > x.min() else 0.01
    pad_y = (y.max() - y.min()) * 0.02 if y.max() > y.min() else 0.01
    axis.set_xlim(x.min() - pad_x, x.max() + pad_x)
    axis.set_ylim(y.min() - pad_y, y.max() + pad_y)

    return m


def plot_potential_grid(grid_x: Sequence[float], grid_y: Sequence[float], potential2D: Sequence[Sequence[float]], axis, *, cmap: str = "viridis", shading: str = "auto", vmin: Optional[float] = None, vmax: Optional[float] = None, **kwargs):
    """Plot a 2D regularly gridded potential array on the provided Axes.

    Parameters
    - grid_x: 1D array-like of x coordinates (length Nx)
    - grid_y: 1D array-like of y coordinates (length Ny)
    - potential2D: 2D array-like with shape (Nx, Ny) containing potential values
    - axis: matplotlib Axes
    - cmap, shading, vmin, vmax: forwarded to pcolormesh/QuadMesh
    - kwargs: additional keyword args forwarded to pcolormesh

    Returns the QuadMesh added to the axis.
    """
    if axis is None:
        raise ValueError("An axis (matplotlib.axes.Axes) must be provided")
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting (install matplotlib)")

    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    Z = np.asarray(potential2D, dtype=float)

    if Z.ndim != 2 or Z.shape != (x.size, y.size):
        raise ValueError("potential2D must be a 2D array with shape (len(grid_x), len(grid_y))")

    # build cell-edge coordinates for pcolormesh
    def edges(coords: np.ndarray) -> np.ndarray:
        if coords.size == 1:
            # single cell: make a tiny edge around the point
            delta = 1.0 if coords[0] == 0 else abs(coords[0]) * 1e-3
            return np.array([coords[0] - delta, coords[0] + delta])
        diffs = np.diff(coords)
        left = coords[0] - diffs[0] / 2.0
        right = coords[-1] + diffs[-1] / 2.0
        centers = coords[:-1] + diffs / 2.0
        return np.concatenate(([left], centers, [right]))

    x_edges = edges(x)
    y_edges = edges(y)

    # pcolormesh expects Z shaped (Ny, Nx) when given 1d edges, so transpose
    m = axis.pcolormesh(x_edges, y_edges, Z.T, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax, **kwargs)

    axis.set_aspect('equal')
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])

    return m
