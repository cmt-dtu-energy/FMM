import numpy as np
import matplotlib.pyplot as plt

def plot_ylog(*y_sets, labels=None, colors=None, sizes=None, alphas=None,
              xvals=None, title=None, xlabel='Index', ylabel='Value (log scale)',
              grid=True, legend=True, **kwargs):
    """
    Create a reusable scatter plot with logarithmic y-axis.

    Parameters
    ----------
    *y_sets : array-like
        One or more 1D arrays or lists of y-values to plot.
    labels : list of str, optional
        Labels for each dataset (for legend).
    colors : list, optional
        Colors for each dataset.
    sizes : list, optional
        Marker sizes for each dataset (default = 5).
    alphas : list, optional
        Transparency for each dataset (default = 0.6).
    xvals : list or array-like, optional
        If provided, should be same length as y_sets or a list of arrays.
        If None, indices (0..N-1) are used for each dataset.
    title : str, optional
        Title of the plot.
    xlabel, ylabel : str, optional
        Axis labels.
    grid : bool, optional
        Whether to show grid lines.
    legend : bool, optional
        Whether to show legend.
    **kwargs :
        Additional keyword arguments passed to plt.scatter().

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))

    n_sets = len(y_sets)
    labels = labels or [f"Set {i}" for i in range(n_sets)]
    colors = colors or plt.cm.tab10(np.arange(n_sets) % 10)
    sizes = sizes or [5] * n_sets

    for i, y in enumerate(y_sets):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("Each y_set must be a 1D array.")
        # determine x-values
        if xvals is None:
            x = np.arange(len(y))
        elif isinstance(xvals, (list, tuple)) and len(xvals) == n_sets:
            x = np.asarray(xvals[i])
        else:
            x = np.asarray(xvals)
        ax.scatter(x, y, color=colors[i], s=sizes[i], label=labels[i], **kwargs)

    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if legend:
        ax.legend()

    plt.tight_layout()
    return fig, ax



def plot_segmented_errors(
    errors_list,
    labels=None,
    title="Relative error per octant (concatenated indices)",
    cmap="tab10",
    point_size=2,
    alpha=0.5,
    yscale="log",
    markerscale=10,
    legend=True,
):
    """
    Plot multiple error arrays back-to-back on the x-axis so they don't overlap.

    Parameters
    ----------
    errors_list : list of 1D array-like
        Each entry is an array of errors for one group (e.g., octant).
    labels : list of str or None
        Legend labels per group. If None, uses 'Octant i'.
    title : str
    cmap : str or Colormap
        Matplotlib colormap name or object used to generate distinct colors.
    point_size : float
        Marker size for scatter points.
    alpha : float
        Point transparency.
    yscale : {"log","linear"}
    markerscale : float
        Legend marker scaling.
    legend : bool
        Whether to draw a legend.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Coerce to numpy arrays and mask non-finite
    errs = [np.asarray(e).ravel() for e in errors_list]
    if labels is None:
        labels = [f"Octant {i+1}" for i in range(len(errs))]

    # Colors
    cmap_obj = plt.get_cmap(cmap, len(errs))
    colors = [cmap_obj(i) for i in range(len(errs))]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    start = 0
    for i, (e, lab, col) in enumerate(zip(errs, labels, colors)):
        # mask invalid or non-positive values if log scale
        mask = np.isfinite(e)
        if yscale == "log":
            mask &= (e > 0)

        e_plot = e[mask]
        if e_plot.size == 0:
            continue

        x = np.arange(start, start + e_plot.size)
        ax.scatter(x, e_plot, s=point_size, alpha=alpha, color=col, label=lab)
        start += e_plot.size

    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel("Target index (concatenated)")
    ax.set_ylabel("Relative error")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if legend:
        ax.legend(markerscale=markerscale, fontsize="small", loc="best")
    fig.tight_layout()
    return fig, ax
