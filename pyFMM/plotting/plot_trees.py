"""Tree plotting utilities for pyFMM.

Provides a simple `plot_tree` dispatcher which selects visualization
based on the number of active dimensions. Currently only 2D plotting
is implemented (draws leaf-node squares).
"""
from typing import Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:  # pragma: no cover - matplotlib may not be available in editor env
    plt = None  # type: ignore
    Rectangle = None  # type: ignore
import numpy as np


def plot_tree(tree, axis: Optional[object] = None, **kwargs) -> Tuple[Optional[object], object]:
    """Dispatch plotting based on the number of active dimensions.

    Parameters
    - tree: an instance of `pyFMM.tree.FMMTree` (or compatible)
    - axis: optional matplotlib Axes; if None a new figure/axes will be created
    - kwargs: forwarded to the specific plot routine (e.g., patch style)

    Returns (fig, ax) when a new figure was created; otherwise (None, ax).
    """
    do_dim = getattr(tree, "do_dimension", [True, True, True])
    n_active = int(sum(bool(x) for x in do_dim))
    if n_active == 2:
        return plot_tree_2D(tree, axis=axis, **kwargs)
    elif n_active == 1:
        raise NotImplementedError("1D tree plotting not implemented yet")
    elif n_active == 3:
        raise NotImplementedError("3D tree plotting not implemented yet")
    else:
        raise ValueError(f"Unsupported number of active dimensions: {n_active}")


def _gather_leaves(tree):
    """Yield leaf nodes by traversing from root.
    This is robust even if `tree.node_list` wasn't populated.
    """
    root = getattr(tree, "root", None)
    if root is None:
        return

    stack = [root]
    while stack:
        node = stack.pop()
        if getattr(node, "is_leaf", False):
            yield node
        else:
            for c in getattr(node, "children", ()):
                if c is not None:
                    stack.append(c)


def plot_tree_2D(tree, axis: Optional[object] = None, edgecolor: str = "k", facecolor: Optional[str] = "none", linewidth: float = 1.0, **kwargs) -> Tuple[Optional[object], object]:
    """Plot a 2D projection of the tree: draw rectangles for every leaf node.

    The two active dimensions are taken from `tree.do_dimension` order (x,y,z).
    Returns (fig, ax) when a new figure was created, else (None, ax).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting (install matplotlib)")

    do_dim = getattr(tree, "do_dimension", [True, True, True])
    active_dims = [i for i, v in enumerate(do_dim) if v]
    if len(active_dims) != 2:
        raise ValueError("plot_tree_2D requires exactly 2 active dimensions on the tree")

    created_fig = False
    if axis is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        ax = axis
        fig = getattr(ax, "figure", None)

    # draw each leaf as a Rectangle
    for node in _gather_leaves(tree):
        c = node.center
        hw = node.half_width
        x0 = c[active_dims[0]] - hw[active_dims[0]]
        y0 = c[active_dims[1]] - hw[active_dims[1]]
        w = 2.0 * hw[active_dims[0]]
        h = 2.0 * hw[active_dims[1]]
        rect = Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, **kwargs)
        ax.add_patch(rect)

    # set equal aspect and reasonable limits
    ax.set_aspect("equal")
    # set limits to tree bounding box if available
    try:
        center = np.asarray(tree.center, dtype=float)
        size = np.asarray(tree.size, dtype=float)
        ax.set_xlim(center[active_dims[0]] - size[active_dims[0]] / 2.0, center[active_dims[0]] + size[active_dims[0]] / 2.0)
        ax.set_ylim(center[active_dims[1]] - size[active_dims[1]] / 2.0, center[active_dims[1]] + size[active_dims[1]] / 2.0)
    except Exception:
        pass

    if created_fig:
        return fig, ax
    return None, ax
