"""
Pure OpenCosmo queries, filters, transforms, and column math.

All functions take a SimulationCollection (or Dataset) and return a transformed one.
No session management, no pandas. Use concat_sims() from oc_data when you need numpy arrays.
"""
import logging

import numpy as np
import opencosmo as oc

from oc_data import concat_sims

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Filtering
# =============================================================================


def query_catalog(ds, min_filters=None, max_filters=None, columns=None):
    """
    Filter a SimulationCollection by column thresholds and optionally select columns.

    Args:
        ds: SimulationCollection from load_catalog().
        min_filters: dict of {column_name: min_value} for lower bounds.
        max_filters: dict of {column_name: max_value} for upper bounds.
        columns: List of columns to select. None keeps all.

    Returns:
        SimulationCollection: Filtered (and optionally column-selected) collection.
    """
    result = ds

    if min_filters:
        for col_name, min_val in min_filters.items():
            result = result.filter(oc.col(col_name) > min_val)

    if max_filters:
        for col_name, max_val in max_filters.items():
            result = result.filter(oc.col(col_name) < max_val)

    if columns:
        result = result.select(columns)

    n_halos = sum(len(v) for v in result.values())
    logger.info(f"Query result: {n_halos} halos")
    return result


def filter_single(ds, column, operator, value):
    """
    Apply a single filter condition using oc.col().

    Args:
        ds: SimulationCollection or Dataset.
        column: Column name to filter on.
        operator: One of '>', '>=', '<', '<=', '==', '!='.
        value: Value to compare against.

    Returns:
        Filtered SimulationCollection/Dataset.
    """
    col_expr = oc.col(column)
    ops = {
        ">": col_expr.__gt__,
        ">=": col_expr.__ge__,
        "<": col_expr.__lt__,
        "<=": col_expr.__le__,
        "==": col_expr.__eq__,
        "!=": col_expr.__ne__,
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator: {operator}. Use one of {list(ops.keys())}")
    condition = ops[operator](value)
    return ds.filter(condition)


def filter_multi(ds, conditions):
    """
    Apply multiple AND filter conditions.

    Args:
        ds: SimulationCollection or Dataset.
        conditions: List of dicts with 'column', 'operator', 'value' keys.

    Returns:
        Filtered SimulationCollection/Dataset.

    Example:
        filtered = filter_multi(ds, [
            {"column": "fof_halo_mass", "operator": ">", "value": 1e14},
            {"column": "sod_halo_T500c", "operator": ">", "value": 3.0},
        ])
    """
    result = ds
    for cond in conditions:
        result = filter_single(result, cond["column"], cond["operator"], cond["value"])
    return result


def sort_and_take(ds, column, n, descending=True):
    """
    Sort by a column and take the top N entries.

    Args:
        ds: SimulationCollection or Dataset.
        column: Column to sort by.
        n: Number of entries to take.
        descending: If True, sort largest first (default).

    Returns:
        SimulationCollection/Dataset with top N entries.

    Note:
        For SimulationCollection, take(n) takes n per simulation, not n total.
        Use concat_sims() afterward if you need a global top-N.
    """
    sorted_ds = ds.sort_by(column, invert=descending)
    return sorted_ds.take(n, at="start")


# =============================================================================
# Column math (add computed columns using oc.col() expressions)
# =============================================================================


def add_column_ratio(ds, output_col, numerator, denominator):
    """
    Add a column that is the ratio of two existing columns.

    Args:
        ds: SimulationCollection or Dataset.
        output_col: Name for the new column.
        numerator: Numerator column name.
        denominator: Denominator column name.

    Returns:
        SimulationCollection/Dataset with new column added.
    """
    expr = oc.col(numerator) / oc.col(denominator)
    return ds.with_new_columns(**{output_col: expr})


def add_column_log10(ds, output_col, column):
    """
    Add a log10 column.

    Args:
        ds: SimulationCollection or Dataset.
        output_col: Name for the new column.
        column: Column to take log10 of.

    Returns:
        SimulationCollection/Dataset with new column added.
    """
    expr = oc.col(column).log10()
    return ds.with_new_columns(**{output_col: expr})


def add_column_norm(ds, output_col, columns):
    """
    Add a Euclidean norm column: sqrt(a^2 + b^2 + ...).

    Args:
        ds: SimulationCollection or Dataset.
        output_col: Name for the new column.
        columns: List of column names to compute norm of.

    Returns:
        SimulationCollection/Dataset with new column added.
    """
    expr = sum(oc.col(c) ** 2 for c in columns) ** 0.5
    return ds.with_new_columns(**{output_col: expr})


def add_column_distance_3d(ds, output_col, ax, ay, az, bx, by, bz, normalize_by=None):
    """
    Add a 3D Euclidean distance column between two coordinate sets.

    distance = sqrt((ax-bx)^2 + (ay-by)^2 + (az-bz)^2)

    Args:
        ds: SimulationCollection or Dataset.
        output_col: Name for the new column.
        ax, ay, az: Column names for first coordinate set.
        bx, by, bz: Column names for second coordinate set.
        normalize_by: Optional column name to divide by.

    Returns:
        SimulationCollection/Dataset with new column added.
    """
    expr = (
        (oc.col(ax) - oc.col(bx)) ** 2
        + (oc.col(ay) - oc.col(by)) ** 2
        + (oc.col(az) - oc.col(bz)) ** 2
    ) ** 0.5
    if normalize_by:
        expr = expr / oc.col(normalize_by)
    return ds.with_new_columns(**{output_col: expr})


def add_column_math(ds, output_col, expr_str):
    """
    Add a column from an arithmetic expression string.

    Supported operations: +, -, *, /, **, log10, sqrt.
    Column references use col('name') syntax.

    Args:
        ds: SimulationCollection or Dataset.
        output_col: Name for the new column.
        expr_str: Expression string, e.g. "col('a') / col('b')".

    Returns:
        SimulationCollection/Dataset with new column added.

    Example:
        ds = add_column_math(ds, "gas_frac", "col('sod_halo_MGas500c') / col('sod_halo_M500c')")
    """
    # Build a safe evaluation context
    def col(name):
        return oc.col(name)

    safe_ns = {"col": col, "np": np}
    expr = eval(expr_str, {"__builtins__": {}}, safe_ns)
    return ds.with_new_columns(**{output_col: expr})


# =============================================================================
# Pre-built derived column sets (from original query_tools.py)
# =============================================================================


def add_velocity_columns(ds):
    """
    Add velocity-derived columns: velocity_mag and v2.

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new columns added.
    """
    v2 = (oc.col("fof_halo_com_vx") ** 2
          + oc.col("fof_halo_com_vy") ** 2
          + oc.col("fof_halo_com_vz") ** 2)

    def compute_vmag(fof_halo_com_vx, fof_halo_com_vy, fof_halo_com_vz):
        return {"velocity_mag": np.sqrt(fof_halo_com_vx**2 + fof_halo_com_vy**2 + fof_halo_com_vz**2)}

    result = ds.with_new_columns(v2=v2)
    result = result.evaluate(compute_vmag, insert=True, vectorize=True)
    return result


def add_offset_columns(ds):
    """
    Add DM-gas center offset columns: dm_gas_offset and dm_gas_offset_ratio.

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new offset columns.
    """
    dx = oc.col("sod_halo_com_x_dm") - oc.col("sod_halo_com_x_gas")
    dy = oc.col("sod_halo_com_y_dm") - oc.col("sod_halo_com_y_gas")
    dz = oc.col("sod_halo_com_z_dm") - oc.col("sod_halo_com_z_gas")
    dm_gas_offset = (dx**2 + dy**2 + dz**2) ** 0.5
    dm_gas_offset_ratio = dm_gas_offset / oc.col("sod_halo_R500c")

    return ds.with_new_columns(
        dm_gas_offset=dm_gas_offset,
        dm_gas_offset_ratio=dm_gas_offset_ratio,
    )


def add_fof_sod_offset(ds):
    """
    Add FoF-SOD center offset columns: fof_sod_offset and fof_sod_offset_ratio.

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new columns.
    """
    dx = oc.col("sod_halo_com_x") - oc.col("fof_halo_center_x")
    dy = oc.col("sod_halo_com_y") - oc.col("fof_halo_center_y")
    dz = oc.col("sod_halo_com_z") - oc.col("fof_halo_center_z")
    offset = (dx**2 + dy**2 + dz**2) ** 0.5
    offset_ratio = offset / oc.col("sod_halo_R200m")

    return ds.with_new_columns(
        fof_sod_offset=offset,
        fof_sod_offset_ratio=offset_ratio,
    )


def add_gas_fraction(ds):
    """
    Add gas/stellar fraction columns at R500c.

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with gas_fraction_500c, hot_gas_fraction_500c, stellar_fraction_500c.
    """
    return ds.with_new_columns(
        gas_fraction_500c=oc.col("sod_halo_MGas500c") / oc.col("sod_halo_M500c"),
        hot_gas_fraction_500c=oc.col("sod_halo_MGasHot500c") / oc.col("sod_halo_M500c"),
        stellar_fraction_500c=oc.col("sod_halo_MStar500c") / oc.col("sod_halo_M500c"),
    )


def add_all_derived(ds):
    """
    Add all standard derived columns: velocity, offsets, gas fractions.

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with all derived columns added.
    """
    ds = add_velocity_columns(ds)
    ds = add_offset_columns(ds)
    ds = add_fof_sod_offset(ds)
    ds = add_gas_fraction(ds)
    return ds


# =============================================================================
# Spatial queries
# =============================================================================


def query_box(ds, min_corner, max_corner):
    """
    Filter dataset to a 3D box region.

    Args:
        ds: SimulationCollection or Dataset.
        min_corner: (x_min, y_min, z_min) tuple.
        max_corner: (x_max, y_max, z_max) tuple.

    Returns:
        Filtered SimulationCollection/Dataset.
    """
    region = oc.make_box(tuple(min_corner), tuple(max_corner))
    return ds.bound(region)


def query_cone(ds, ra, dec, radius_arcmin):
    """
    Filter lightcone dataset to a cone on the sky.

    Args:
        ds: SimulationCollection or Dataset (must be lightcone data).
        ra: Right Ascension of cone center (degrees).
        dec: Declination of cone center (degrees).
        radius_arcmin: Cone radius in arcminutes.

    Returns:
        Filtered SimulationCollection/Dataset.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    center = SkyCoord(ra * u.deg, dec * u.deg)
    region = oc.make_cone(center, radius_arcmin * u.arcmin)
    return ds.bound(region)


# =============================================================================
# Composite analysis (merger candidates)
# =============================================================================


def find_merger_candidates(ds, n_candidates=10, min_mass=1e14):
    """
    Identify merger candidates using composite scoring of dynamical indicators.

    Scores based on: DM-gas offset, velocity magnitude, velocity dispersion,
    temperature excess relative to mass, and low concentration.

    Args:
        ds: SimulationCollection.
        n_candidates: Number of top candidates to return.
        min_mass: Minimum M500c threshold (Msun/h).

    Returns:
        dict: {column_name: np.ndarray} for the top merger candidates,
              including a 'merger_score' column.
    """
    ds_aug = add_velocity_columns(ds)
    ds_aug = add_offset_columns(ds_aug)
    ds_aug = ds_aug.filter(oc.col("sod_halo_M500c") > min_mass)
    ds_aug = ds_aug.filter(oc.col("sod_halo_T500c") > 0)
    ds_aug = ds_aug.filter(oc.col("sod_halo_R500c") > 0)

    cols = ["unique_tag", "fof_halo_tag", "sod_halo_M500c", "sod_halo_T500c",
            "sod_halo_cdelta", "fof_halo_1D_vel_disp",
            "dm_gas_offset_ratio", "velocity_mag"]
    data = concat_sims(ds_aug, columns=cols)

    offset_r = data["dm_gas_offset_ratio"]
    v_mag = data["velocity_mag"]
    vel_disp = data["fof_halo_1D_vel_disp"]
    temp = data["sod_halo_T500c"]
    mass = data["sod_halo_M500c"]
    conc = data["sod_halo_cdelta"]

    expected_temp = (mass / 1e15) ** (2. / 3.) * 5.0
    temp_excess = temp / np.clip(expected_temp, 1e-3, None)

    score = (
        2.0 * np.log10(np.clip(offset_r, 1e-6, None))
        + 1.5 * np.log10(np.clip(v_mag / 500.0, 1e-6, None))
        + 1.0 * np.log10(np.clip(vel_disp / 1000.0, 1e-6, None))
        + 1.0 * np.log10(np.clip(temp_excess, 1e-6, None))
        + 0.5 * np.log10(np.clip(1.0 / (conc + 1.0), 1e-6, None))
        + 0.5 * np.log10(np.clip(mass / 1e15, 1e-6, None))
    )

    top_idx = np.argsort(score)[-n_candidates:][::-1]
    result = {col: data[col][top_idx] for col in cols}
    result["merger_score"] = score[top_idx]
    return result
