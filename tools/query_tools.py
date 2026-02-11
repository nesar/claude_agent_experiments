"""
Query and filtering tools for HACC halo catalogs.

Works with SimulationCollection objects (from oc.open on SCIDAC files).
Uses oc.col() arithmetic and the evaluate() API for derived columns.
"""
import logging
import numpy as np
import opencosmo as oc

from data_tools import DEFAULT_CATALOG, load_catalog, concat_sims

logging.basicConfig(level=logging.INFO)


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

    Example:
        filtered = query_catalog(ds,
            min_filters={"fof_halo_mass": 1e14, "sod_halo_T500c": 3.0},
            max_filters={"sod_halo_cdelta": 20.0},
            columns=["unique_tag", "fof_halo_mass", "sod_halo_T500c"])
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
    logging.info(f"Query result: {n_halos} halos")
    return result


def add_velocity_columns(ds):
    """
    Add velocity-derived columns using OpenCosmo column arithmetic.

    Adds:
        - velocity_mag: sqrt(vx^2 + vy^2 + vz^2) from FoF COM velocities
        - v2: vx^2 + vy^2 + vz^2

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new columns added.
    """
    v2 = (oc.col("fof_halo_com_vx")**2
          + oc.col("fof_halo_com_vy")**2
          + oc.col("fof_halo_com_vz")**2)

    def compute_vmag(fof_halo_com_vx, fof_halo_com_vy, fof_halo_com_vz):
        return {"velocity_mag": np.sqrt(fof_halo_com_vx**2 + fof_halo_com_vy**2 + fof_halo_com_vz**2)}

    result = ds.with_new_columns(v2=v2)
    result = result.evaluate(compute_vmag, insert=True, vectorize=True)
    return result


def add_offset_columns(ds):
    """
    Add DM-gas center offset columns using OpenCosmo column arithmetic.

    Adds:
        - dm_gas_offset: 3D distance between DM and gas centers (Mpc/h)
        - dm_gas_offset_ratio: offset normalized by R500c

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new offset columns.
    """
    dx = oc.col("sod_halo_com_x_dm") - oc.col("sod_halo_com_x_gas")
    dy = oc.col("sod_halo_com_y_dm") - oc.col("sod_halo_com_y_gas")
    dz = oc.col("sod_halo_com_z_dm") - oc.col("sod_halo_com_z_gas")
    dm_gas_offset = (dx**2 + dy**2 + dz**2)**0.5
    dm_gas_offset_ratio = dm_gas_offset / oc.col("sod_halo_R500c")

    return ds.with_new_columns(
        dm_gas_offset=dm_gas_offset,
        dm_gas_offset_ratio=dm_gas_offset_ratio
    )


def add_fof_sod_offset(ds):
    """
    Add FoF-SOD center offset columns.

    Adds:
        - fof_sod_offset: 3D distance between FoF center and SOD COM
        - fof_sod_offset_ratio: offset normalized by R200m

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new columns.
    """
    dx = oc.col("sod_halo_com_x") - oc.col("fof_halo_center_x")
    dy = oc.col("sod_halo_com_y") - oc.col("fof_halo_center_y")
    dz = oc.col("sod_halo_com_z") - oc.col("fof_halo_center_z")
    offset = (dx**2 + dy**2 + dz**2)**0.5
    offset_ratio = offset / oc.col("sod_halo_R200m")

    return ds.with_new_columns(
        fof_sod_offset=offset,
        fof_sod_offset_ratio=offset_ratio
    )


def add_gas_fraction(ds):
    """
    Add gas fraction columns using OpenCosmo column arithmetic.

    Adds:
        - gas_fraction_500c: MGas500c / M500c
        - hot_gas_fraction_500c: MGasHot500c / M500c
        - stellar_fraction_500c: MStar500c / M500c

    Args:
        ds: SimulationCollection.

    Returns:
        SimulationCollection with new fraction columns.
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
    # Add derived columns and filter
    ds_aug = add_velocity_columns(ds)
    ds_aug = add_offset_columns(ds_aug)
    ds_aug = ds_aug.filter(oc.col("sod_halo_M500c") > min_mass)
    ds_aug = ds_aug.filter(oc.col("sod_halo_T500c") > 0)
    ds_aug = ds_aug.filter(oc.col("sod_halo_R500c") > 0)

    # Concatenate across sims for scoring
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

    # Expected T-M relation: T ~ (M/1e15)^(2/3) * 5 keV
    expected_temp = (mass / 1e15)**(2./3.) * 5.0
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
