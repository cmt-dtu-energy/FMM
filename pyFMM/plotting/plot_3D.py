import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go



def plot_points_3d(*point_sets, labels=None, colors=None, sizes=None,
                   bounds=None, title=None, equal_aspect=True, **kwargs):
    """
    Create a reusable 3D scatter plot of one or more point sets.

    Parameters
    ----------
    *point_sets : arrays of shape (N, 3)
        One or more arrays of 3D coordinates to plot.
    labels : list of str, optional
        Labels for each point set (used in legend).
    colors : list, optional
        Colors for each point set.
    sizes : list, optional
        Marker sizes for each point set (default = 5).
    bounds : tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax)), optional
        Axis bounds. If None, inferred from all point sets.
    title : str, optional
        Plot title.
    equal_aspect : bool, optional
        Whether to enforce equal scaling for all axes (default: True).
    **kwargs :
        Additional keyword arguments passed to `ax.scatter()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis object.
    """
    # --- create figure and 3D axis ---
    fig = plt.figure(figsize=kwargs.pop('figsize', (10, 8)))
    ax = fig.add_subplot(111, projection='3d')

    # --- normalise optional inputs ---
    n_sets = len(point_sets)
    labels = labels or [f"Set {i}" for i in range(n_sets)]
    colors = colors or plt.cm.tab10(np.arange(n_sets) % 10)
    sizes = sizes or [5] * n_sets

    # --- plot each point set ---
    for i, pts in enumerate(point_sets):
        if pts is None or len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=colors[i],
            s=sizes[i],
            label=labels[i],
            **kwargs
        )

    # --- axis labels and title ---
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    if title:
        ax.set_title(title)

    # --- set bounds automatically ---
    if bounds is None:
        all_pts = np.vstack([p for p in point_sets if p is not None])
        x_min, y_min, z_min = np.min(all_pts, axis=0)
        x_max, y_max, z_max = np.max(all_pts, axis=0)
        # add a small margin
        margin = 0.05 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
    else:
        (xlim, ylim, zlim) = bounds
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

    # --- enforce equal aspect if requested ---
    if equal_aspect:
        _set_equal_aspect_3d(ax)

    ax.legend()
    plt.tight_layout()

    return fig, ax


def _set_equal_aspect_3d(ax):
    """Set equal aspect ratio for a 3D axis."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    max_range = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])


def plot_points_3d_plotly(*point_sets, labels=None, colors=None, sizes=None,
                          alphas=None, bounds=None, title=None, equal_aspect=True):
    """
    Interactive 3D scatter using Plotly. Returns a plotly.graph_objects.Figure.
    """
    n_sets = len(point_sets)
    labels = labels or [f"Set {i}" for i in range(n_sets)]
    # use Plotly's default cycle if colors not given
    default_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
                      '#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    colors = colors or [default_colors[i % len(default_colors)] for i in range(n_sets)]
    sizes = sizes or [2] * n_sets
    alphas = alphas or [0.6] * n_sets

    fig = go.Figure()

    for i, pts in enumerate(point_sets):
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts)
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            name=labels[i],
            marker=dict(size=sizes[i], color=colors[i], opacity=alphas[i])
        ))

    # bounds / aspect
    if bounds is None and n_sets > 0:
        all_pts = np.vstack([p for p in point_sets if p is not None and len(p) > 0])
        mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
        rng = (maxs - mins).max()
        mid = (mins + maxs) / 2.0
        xlim = (mid[0]-rng/2, mid[0]+rng/2)
        ylim = (mid[1]-rng/2, mid[1]+rng/2)
        zlim = (mid[2]-rng/2, mid[2]+rng/2)
    else:
        (xlim, ylim, zlim) = bounds

    fig.update_layout(
        title=title or "",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis",
            xaxis=dict(range=list(xlim)),
            yaxis=dict(range=list(ylim)),
            zaxis=dict(range=list(zlim)),
            aspectmode='cube' if equal_aspect else 'data'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing='constant')
    )
    return fig