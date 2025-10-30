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


def plot_neighbors_node_on_axis(node, axis, node_color: str = "green", neighbor_color: str = "red", alpha: float = 0.3, **kwargs):
    """Highlight `node` (green by default) and its near neighbors (red) on the provided axis.

    This function always requires an `axis` argument and will not create a new figure.
    Style options for the patches can be provided via kwargs.
    """
    if axis is None:
        raise ValueError("An axis must be provided; this function does not create figures")
    # determine active dims from the tree attached to node
    tree = getattr(node, 'tree', None)
    if tree is None:
        raise ValueError("Provided node does not reference a parent tree via node.tree")
    do_dim = getattr(tree, 'do_dimension', [True, True, True])
    active_dims = [i for i, v in enumerate(do_dim) if v]
    if len(active_dims) != 2:
        raise NotImplementedError("Neighbor plotting only implemented for 2 active dimensions")

    # helper to draw a node rectangle
    def _draw_node(n, color):
        c = np.asarray(n.center, dtype=float)
        hw = np.asarray(n.half_width, dtype=float)
        x0 = c[active_dims[0]] - hw[active_dims[0]]
        y0 = c[active_dims[1]] - hw[active_dims[1]]
        w = 2.0 * hw[active_dims[0]]
        h = 2.0 * hw[active_dims[1]]
        rect = Rectangle((x0, y0), w, h, edgecolor=color, facecolor=color, alpha=alpha, **kwargs)
        axis.add_patch(rect)

    # draw main node
    _draw_node(node, node_color)
    # draw neighbors
    for n in getattr(node, 'neighbors', ()):  # type: ignore
        if n is node:
            continue
        _draw_node(n, neighbor_color)


def plot_neighbors_point_on_axis(point, tree, axis, node_color: str = "green", neighbor_color: str = "red", alpha: float = 0.3, **kwargs):
    """Find the leaf containing `point` and highlight it and its neighbors on `axis`.

    `point` may be a float (1D), sequence of 2 (2D) or 3 (3D) numbers. The function
    will validate that the number of provided coordinates matches the number of active dimensions.
    """
    if axis is None:
        raise ValueError("An axis must be provided; this function does not create figures")
    do_dim = np.asarray(getattr(tree, 'do_dimension', [True, True, True]), dtype=bool)
    n_active = int(do_dim.sum())
    p = np.asarray(point, dtype=float)
    if p.ndim == 0:
        p = np.asarray([p])
    if not (p.size == n_active or p.size == 3):
        raise ValueError(f"Point dimensionality ({p.size}) does not match number of active dimensions ({n_active})")

    node = tree.find_leaf_for_point(p)
    if node is None:
        return None
    plot_neighbors_node_on_axis(node, axis, node_color=node_color, neighbor_color=neighbor_color, alpha=alpha, **kwargs)
    return node


def plot_tree_2D(tree, axis: Optional[object] = None, **kwargs) -> Tuple[Optional[object], object]:
    """Plot a 2D projection of the tree: draw rectangles for every leaf node.

    All drawing/style options (edgecolor, facecolor, linewidth, alpha, etc.)
    should be provided via `kwargs`. Defaults are applied when not provided.

    The two active dimensions are taken from `tree.do_dimension` order (x,y,z).
    Returns (fig, ax) when a new figure was created, else (None, ax).
    """
    # style defaults (consumed from kwargs)
    edgecolor = kwargs.pop("edgecolor", "k")
    facecolor = kwargs.pop("facecolor", "none")
    linewidth = kwargs.pop("linewidth", 1.0)
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

    # set axis labels based on active dimensions ('x','y','z')
    labels = ["x", "y", "z"]
    ax.set_xlabel(labels[active_dims[0]])
    ax.set_ylabel(labels[active_dims[1]])

    if created_fig:
        return fig, ax
    return None, ax
