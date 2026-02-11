"""
Statistical analysis tools for HACC halo catalogs.

Works with SimulationCollection objects. Uses concat_sims() for cross-sim
operations and oc.evaluate() for per-halo computations.
"""
import logging
import numpy as np
import opencosmo as oc

from data_tools import concat_sims

logging.basicConfig(level=logging.INFO)


def compute_stats(ds, columns):
    """
    Compute basic statistics for specified columns across all simulations.

    Args:
        ds: SimulationCollection.
        columns: str or list of column names.

    Returns:
        dict: {column_name: {count, mean, median, std, min, max, q25, q75}}.
    """
    if isinstance(columns, str):
        columns = [columns]

    data = concat_sims(ds, columns=columns)
    results = {}
    for col in columns:
        x = data[col]
        valid = x[np.isfinite(x)]
        if len(valid) == 0:
            results[col] = {"error": "no valid values"}
            continue
        results[col] = {
            "count": int(len(valid)),
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "q25": float(np.percentile(valid, 25)),
            "q75": float(np.percentile(valid, 75)),
        }
    return results


def find_extrema(ds, column, n=10, mode="max"):
    """
    Find the top N halos by a given column across all simulations.

    Args:
        ds: SimulationCollection.
        column: Column name to rank by.
        n: Number of extrema to return.
        mode: 'max' for largest, 'min' for smallest.

    Returns:
        dict: {column_name: np.ndarray} for the top N halos.
    """
    sorted_ds = ds.sort_by(column, invert=(mode == "max"))
    top_ds = sorted_ds.take(n, at="start")

    # Concatenate and re-sort across sims
    all_cols = list(next(iter(top_ds.values())).columns)
    data = concat_sims(top_ds, columns=all_cols)

    idx = np.argsort(data[column])
    if mode == "max":
        idx = idx[::-1]
    idx = idx[:n]

    return {col: data[col][idx] for col in all_cols}


def find_outliers(ds, column, method="iqr", threshold=3.0):
    """
    Detect outliers in a column across all simulations.

    Args:
        ds: SimulationCollection.
        column: Column to analyze.
        method: 'iqr', 'zscore', or 'modified_zscore'.
        threshold: Sensitivity (1.5 for IQR, 3.0 for z-score).

    Returns:
        dict with keys:
            - 'high': indices of high outliers
            - 'low': indices of low outliers
            - 'values': the column values
            - 'thresholds': (lower, upper) bounds
    """
    data = concat_sims(ds, columns=[column])
    x = data[column]
    valid_mask = np.isfinite(x)
    x_valid = x[valid_mask]

    if len(x_valid) == 0:
        return {"high": np.array([]), "low": np.array([]), "values": x, "thresholds": (0, 0)}

    if method == "iqr":
        q1, q3 = np.percentile(x_valid, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == "zscore":
        mu, sigma = np.mean(x_valid), np.std(x_valid)
        lower = mu - threshold * sigma
        upper = mu + threshold * sigma
    elif method == "modified_zscore":
        med = np.median(x_valid)
        mad = np.median(np.abs(x_valid - med))
        if mad == 0:
            return {"high": np.array([]), "low": np.array([]), "values": x, "thresholds": (med, med)}
        scale = threshold / 0.6745 * mad
        lower = med - scale
        upper = med + scale
    else:
        raise ValueError(f"Unknown method: {method}")

    high_idx = np.where(valid_mask & (x > upper))[0]
    low_idx = np.where(valid_mask & (x < lower))[0]

    return {
        "high": high_idx,
        "low": low_idx,
        "n_high": len(high_idx),
        "n_low": len(low_idx),
        "values": x,
        "thresholds": (float(lower), float(upper)),
    }


def compute_histogram(ds, column, bins=50, log=False, range=None):
    """
    Compute a histogram of a column across all simulations.

    Args:
        ds: SimulationCollection.
        column: Column name.
        bins: Number of bins or array of bin edges.
        log: If True, histogram in log10 space.
        range: (min, max) tuple for bin range.

    Returns:
        dict: {counts, bin_edges, bin_centers}.
    """
    data = concat_sims(ds, columns=[column])
    x = data[column]
    x = x[np.isfinite(x)]

    if log:
        x = x[x > 0]
        x = np.log10(x)

    counts, bin_edges = np.histogram(x, bins=bins, range=range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {"counts": counts, "bin_edges": bin_edges, "bin_centers": bin_centers}


def compute_scaling_relation(ds, col_x, col_y, log_x=True, log_y=True, n_bins=20):
    """
    Compute a binned scaling relation (median + scatter) between two columns.

    Args:
        ds: SimulationCollection.
        col_x: X-axis column name.
        col_y: Y-axis column name.
        log_x: If True, bin in log10(x) space.
        log_y: If True, compute stats in log10(y) space.
        n_bins: Number of bins.

    Returns:
        dict: {x_centers, y_median, y_q16, y_q84, x_raw, y_raw}.
    """
    data = concat_sims(ds, columns=[col_x, col_y])
    x = data[col_x]
    y = data[col_y]

    # Filter valid
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if log_x:
        x = np.log10(x)
    if log_y:
        y = np.log10(y)

    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y_median = np.full(n_bins, np.nan)
    y_q16 = np.full(n_bins, np.nan)
    y_q84 = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.sum(in_bin) > 2:
            y_median[i] = np.median(y[in_bin])
            y_q16[i] = np.percentile(y[in_bin], 16)
            y_q84[i] = np.percentile(y[in_bin], 84)

    return {
        "x_centers": bin_centers,
        "y_median": y_median,
        "y_q16": y_q16,
        "y_q84": y_q84,
        "x_raw": x,
        "y_raw": y,
    }
