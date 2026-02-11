"""
Visualization tools for HACC halo catalogs.

Creates publication-quality matplotlib plots from SimulationCollection data.
Includes particle projection visualization when particle data is available.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import opencosmo as oc

from data_tools import concat_sims
from analysis_tools import compute_scaling_relation

logging.basicConfig(level=logging.INFO)


def plot_distribution(ds, column, bins=50, log=False, ax=None, save_path=None, **hist_kwargs):
    """
    Plot a 1D histogram of a column across all simulations.

    Args:
        ds: SimulationCollection.
        column: Column name.
        bins: Number of bins.
        log: If True, plot log10 of the column.
        ax: Optional matplotlib Axes. Creates new figure if None.
        save_path: If provided, save figure to this path.
        **hist_kwargs: Additional kwargs for plt.hist.

    Returns:
        matplotlib.axes.Axes
    """
    data = concat_sims(ds, columns=[column])
    x = data[column]
    x = x[np.isfinite(x)]

    if log:
        x = x[x > 0]
        x = np.log10(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    defaults = {"bins": bins, "alpha": 0.7, "edgecolor": "black", "color": "steelblue"}
    defaults.update(hist_kwargs)
    ax.hist(x, **defaults)

    xlabel = f"log10({column})" if log else column
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of {column} (N={len(x)})", fontsize=13)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved to {save_path}")

    return ax


def plot_scatter(ds, col_x, col_y, log_x=False, log_y=False,
                 color_by=None, log_color=False,
                 ax=None, save_path=None, **scatter_kwargs):
    """
    Scatter plot of two columns across all simulations.

    Args:
        ds: SimulationCollection.
        col_x: X-axis column name.
        col_y: Y-axis column name.
        log_x: If True, plot log10(x).
        log_y: If True, plot log10(y).
        color_by: Optional column name for color mapping.
        log_color: If True, take log10 of color values.
        ax: Optional matplotlib Axes.
        save_path: If provided, save figure to this path.
        **scatter_kwargs: Additional kwargs for plt.scatter.

    Returns:
        matplotlib.axes.Axes
    """
    cols = [col_x, col_y]
    if color_by:
        cols.append(color_by)
    data = concat_sims(ds, columns=cols)

    x = data[col_x]
    y = data[col_y]
    mask = np.isfinite(x) & np.isfinite(y)

    if color_by:
        c = data[color_by]
        mask &= np.isfinite(c)

    if log_x or log_y:
        if log_x:
            mask &= (x > 0)
        if log_y:
            mask &= (y > 0)

    x, y = x[mask], y[mask]
    if log_x:
        x = np.log10(x)
    if log_y:
        y = np.log10(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    defaults = {"s": 5, "alpha": 0.6}
    defaults.update(scatter_kwargs)

    if color_by:
        c = data[color_by][mask]
        if log_color:
            c = c[c > 0] if np.any(c <= 0) else c
            c = np.log10(c)
        sc = ax.scatter(x, y, c=c, cmap="viridis", **defaults)
        clabel = f"log10({color_by})" if log_color else color_by
        plt.colorbar(sc, ax=ax, label=clabel)
    else:
        ax.scatter(x, y, **defaults)

    xlabel = f"log10({col_x})" if log_x else col_x
    ylabel = f"log10({col_y})" if log_y else col_y
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{col_y} vs {col_x} (N={len(x)})", fontsize=13)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved to {save_path}")

    return ax


def plot_scaling_relation(ds, col_x, col_y, log_x=True, log_y=True,
                          n_bins=20, show_scatter=True,
                          ax=None, save_path=None):
    """
    Plot a binned scaling relation with median and 16/84 percentile scatter.

    Args:
        ds: SimulationCollection.
        col_x: X-axis column name.
        col_y: Y-axis column name.
        log_x: If True, bin in log10(x).
        log_y: If True, compute stats in log10(y).
        n_bins: Number of bins.
        show_scatter: If True, also plot individual data points.
        ax: Optional matplotlib Axes.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib.axes.Axes
    """
    rel = compute_scaling_relation(ds, col_x, col_y,
                                   log_x=log_x, log_y=log_y, n_bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    if show_scatter:
        ax.scatter(rel["x_raw"], rel["y_raw"], s=2, alpha=0.2, color="gray", zorder=1)

    valid = np.isfinite(rel["y_median"])
    ax.plot(rel["x_centers"][valid], rel["y_median"][valid],
            "o-", color="crimson", linewidth=2, markersize=5, zorder=3, label="Median")
    ax.fill_between(rel["x_centers"][valid], rel["y_q16"][valid], rel["y_q84"][valid],
                    alpha=0.25, color="crimson", zorder=2, label="16-84%")

    xlabel = f"log10({col_x})" if log_x else col_x
    ylabel = f"log10({col_y})" if log_y else col_y
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{col_y} vs {col_x}", fontsize=13)
    ax.legend(fontsize=10)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved to {save_path}")

    return ax


def plot_multi_distribution(ds, columns, log_columns=None, ncols=3, save_path=None):
    """
    Plot histograms of multiple columns in a grid layout.

    Args:
        ds: SimulationCollection.
        columns: List of column names.
        log_columns: List of column names to plot in log10 space.
        ncols: Number of columns in the grid.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib.figure.Figure
    """
    log_columns = log_columns or []
    n = len(columns)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(columns):
        row, c = divmod(i, ncols)
        ax = axes[row, c]
        use_log = col in log_columns
        plot_distribution(ds, col, log=use_log, ax=ax)

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, c = divmod(i, ncols)
        axes[row, c].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Particle-level visualization (requires particle HDF5 files)
# ---------------------------------------------------------------------------

def visualize_halo_quick(data, halo_id=None, save_path=None):
    """
    Create a 2x2 particle projection plot for a single halo using
    opencosmo.analysis.visualize_halo.

    Requires a StructureCollection loaded with both halo properties and
    particle data (via oc.open("haloproperties.hdf5", "haloparticles.hdf5")).

    Args:
        data: StructureCollection (from oc.open with two files, or load_with_particles).
        halo_id: Unique halo tag to visualize. If None, picks one at random.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib.figure.Figure

    Example:
        with oc.open("haloproperties.hdf5", "haloparticles.hdf5") as data:
            fig = visualize_halo_quick(data)
    """
    from opencosmo.analysis import visualize_halo

    if halo_id is None:
        sample = data.take(1, at="random")
        halo = next(sample.halos())
        halo_id = halo["halo_properties"]["unique_tag"]
        logging.info(f"Selected random halo: {halo_id}")

    fig = visualize_halo(halo_id, data)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved halo visualization to {save_path}")

    return fig


def visualize_halo_array(data, halo_ids, field=("dm", "particle_mass"),
                         width=6.0, save_path=None):
    """
    Create a grid of particle projections for multiple halos using
    opencosmo.analysis.halo_projection_array.

    Args:
        data: StructureCollection with particle data.
        halo_ids: 2D array-like of halo unique_tags defining the grid layout.
                  E.g., np.array([[id1, id2], [id3, id4]]) for a 2x2 grid.
        field: Tuple of (particle_type, field_name) to project.
               Options: ("dm", "particle_mass"), ("gas", "particle_mass"),
                        ("star", "particle_mass"), ("gas", "temperature").
        width: Width of each projection in units of R_halo.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib.figure.Figure

    Example:
        with oc.open("haloproperties.hdf5", "haloparticles.hdf5") as data:
            ids = np.array([[id1, id2], [id3, id4]])
            fig = visualize_halo_array(data, ids, field=("gas", "particle_mass"))
    """
    from opencosmo.analysis import halo_projection_array

    halo_ids = np.atleast_2d(halo_ids)
    fig = halo_projection_array(halo_ids, data, field=field, width=width)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved halo array to {save_path}")

    return fig


def visualize_halo_multifield(data, halo_ids, save_path=None):
    """
    Create a multi-field projection array: DM + gas mass for each halo.

    Args:
        data: StructureCollection with particle data.
        halo_ids: 2D array-like of halo unique_tags (rows = halos, cols = repeated).
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib.figure.Figure

    Example:
        with oc.open("haloproperties.hdf5", "haloparticles.hdf5") as data:
            ids = np.array([[id1], [id2], [id3]])
            fig = visualize_halo_multifield(data, ids)
    """
    from opencosmo.analysis import halo_projection_array

    halo_ids = np.atleast_2d(halo_ids)

    params = {
        "fields": (
            [("dm", "particle_mass"), ("gas", "particle_mass")],
        ) * halo_ids.shape[0],
        "labels": (
            ["Dark Matter", "Gas"],
        ) * halo_ids.shape[0],
        "cmaps": (
            ["gray", "cividis"],
        ) * halo_ids.shape[0],
    }

    fig = halo_projection_array(halo_ids, data, params=params, length_scale="all left")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved multi-field array to {save_path}")

    return fig
