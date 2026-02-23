"""
OpenCosmo radial profile tools for HACC halo catalogs.

Handles 2D profile arrays (n_halos x n_bins) for quantities like
temperature, entropy, density, etc. Adapted from MACosmo's ProfilesWrapper.
"""
import logging
import re

import h5py
import numpy as np
import opencosmo as oc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfilesWrapper:
    """
    Wrapper for halo profile data (2D arrays: n_halos x n_bins).

    Profiles are stored as numpy arrays since they're inherently 2D.
    """

    def __init__(self, data, n_bins):
        """
        Args:
            data: dict of {column_name: np.ndarray(n_halos, n_bins)}.
            n_bins: Number of radial bins.
        """
        self._data = data
        self._n_bins = n_bins
        self._columns = list(data.keys())
        self._n_halos = data[self._columns[0]].shape[0] if self._columns else 0

    @property
    def columns(self):
        return self._columns

    @property
    def n_bins(self):
        return self._n_bins

    @property
    def n_halos(self):
        return self._n_halos

    def __len__(self):
        return self._n_halos

    def get_column(self, column):
        """Get a profile column as 2D array (n_halos, n_bins)."""
        if column not in self._data:
            raise KeyError(f"Column '{column}' not found. Available: {self._columns}")
        return self._data[column]

    def get_profile(self, index, columns=None):
        """
        Get profile data for a single halo by index.

        Args:
            index: Halo index (0-based).
            columns: List of columns to include (default: all).

        Returns:
            dict: {column_name: np.ndarray(n_bins,)}.
        """
        if index < 0 or index >= self._n_halos:
            raise IndexError(f"Index {index} out of range [0, {self._n_halos})")
        cols = columns or self._columns
        return {col: self._data[col][index] for col in cols if col in self._data}

    def get_profiles_by_mask(self, mask):
        """
        Get profiles for halos matching a boolean mask.

        Args:
            mask: Boolean np.ndarray of shape (n_halos,).

        Returns:
            ProfilesWrapper with filtered profiles.
        """
        filtered_data = {col: arr[mask] for col, arr in self._data.items()}
        return ProfilesWrapper(filtered_data, self._n_bins)

    def get_profiles_by_indices(self, indices):
        """
        Get profiles for halos at specific indices.

        Args:
            indices: Integer np.ndarray of indices.

        Returns:
            ProfilesWrapper with selected profiles.
        """
        filtered_data = {col: arr[indices] for col, arr in self._data.items()}
        return ProfilesWrapper(filtered_data, self._n_bins)

    def compute_median_profile(self, column):
        """Compute median profile across all halos for a given column."""
        return np.median(self._data[column], axis=0)

    def compute_percentile_profiles(self, column, percentiles=None):
        """
        Compute percentile profiles (e.g., 16th, 50th, 84th for 1-sigma range).

        Args:
            column: Profile column name.
            percentiles: List of percentiles (default: [16, 50, 84]).

        Returns:
            dict: {"p16": array, "p50": array, "p84": array, ...}
        """
        if percentiles is None:
            percentiles = [16, 50, 84]
        result = {}
        for p in percentiles:
            result[f"p{int(p)}"] = np.percentile(self._data[column], p, axis=0)
        return result


# =============================================================================
# Loading functions
# =============================================================================


def _is_scidac_format(f):
    """Check if file uses SCIDAC partitioning."""
    keys = list(f.keys())
    if not keys:
        return False
    scidac_pattern = re.compile(r'^SCIDAC_\d{3}$')
    return all(scidac_pattern.match(k) for k in keys)


def load_profiles(path):
    """
    Load halo radial profiles from an HDF5 file.

    Handles both SCIDAC-partitioned and Frontier-style files.

    Args:
        path: Path to HDF5 file containing halo_profiles data.

    Returns:
        ProfilesWrapper: Wrapper with 2D profile arrays.

    Common profile columns:
        - sod_halo_bin_radius: Radial bin centers
        - sod_halo_bin_gas_entropy: Gas entropy profile
        - sod_halo_bin_gas_temperature: Gas temperature profile
        - sod_halo_bin_gas_ne: Gas electron density profile
    """
    with h5py.File(path, 'r') as f:
        if _is_scidac_format(f):
            return _load_scidac_profiles(f)

    # Try opencosmo for non-SCIDAC files (e.g., Frontier)
    try:
        ds = oc.open(path)
        # If it's a multi-table file, get halo_profiles table
        if hasattr(ds, 'keys') and not hasattr(ds, 'columns'):
            sub_names = list(ds.keys())
            if 'halo_profiles' not in sub_names:
                raise ValueError(f"No halo_profiles table in file. Available: {sub_names}")
            ds = ds['halo_profiles']

        columns = list(ds.columns)
        raw = ds.get_data(output="numpy")

        # Determine n_bins from shape of first column
        first_arr = raw[columns[0]]
        if first_arr.ndim == 2:
            n_bins = first_arr.shape[1]
        else:
            raise ValueError("Profile data must be 2D (n_halos x n_bins)")

        data = {col: raw[col] for col in columns}
        logger.info(f"Loaded profiles: {first_arr.shape[0]} halos, {n_bins} bins, {len(columns)} columns")
        return ProfilesWrapper(data, n_bins)

    except Exception as e:
        raise ValueError(f"Failed to load profiles from {path}: {e}")


def _load_scidac_profiles(f):
    """Load profiles from an open SCIDAC-format HDF5 file handle."""
    block_names = sorted([k for k in f.keys() if k.startswith('SCIDAC_')])

    if not block_names:
        raise ValueError("No SCIDAC blocks found in file")

    first_block = f[block_names[0]]
    if 'halo_profiles' not in first_block:
        raise ValueError("No halo_profiles dataset found in file")

    data_group = first_block['halo_profiles']['data']
    columns = [k for k in data_group.keys() if isinstance(data_group[k], h5py.Dataset)]
    n_bins = data_group[columns[0]].shape[1]

    all_data = {col: [] for col in columns}
    for block_name in block_names:
        data_group = f[f'{block_name}/halo_profiles/data']
        for col in columns:
            all_data[col].append(data_group[col][:])

    data = {col: np.concatenate(arrs) for col, arrs in all_data.items()}
    n_halos = data[columns[0]].shape[0]
    logger.info(f"Loaded SCIDAC profiles: {n_halos} halos, {n_bins} bins, {len(columns)} columns from {len(block_names)} blocks")
    return ProfilesWrapper(data, n_bins)


# =============================================================================
# Analysis functions
# =============================================================================


def get_halo_profile(profiles, index, columns=None):
    """
    Get the radial profile for a single halo by index.

    Args:
        profiles: ProfilesWrapper from load_profiles().
        index: Index of the halo (0-based).
        columns: List of columns to include (default: all).

    Returns:
        dict: {column_name: np.ndarray(n_bins,)}.
    """
    return profiles.get_profile(index, columns)


def compute_median_profiles(profiles, columns, mask=None, percentiles=None):
    """
    Compute median (and percentile) profiles across halos.

    Args:
        profiles: ProfilesWrapper from load_profiles().
        columns: List of profile column names to compute medians for.
        mask: Optional boolean np.ndarray to select a subset of halos.
        percentiles: List of percentiles (default: [16, 50, 84]).

    Returns:
        dict: {
            "n_halos_used": int,
            "n_bins": int,
            "radius_bins": list (if available),
            "profiles": {col_name: {"p16": [...], "p50": [...], "p84": [...]}}
        }
    """
    if percentiles is None:
        percentiles = [16, 50, 84]

    if mask is not None:
        profiles = profiles.get_profiles_by_mask(mask)

    result = {
        "n_halos_used": profiles.n_halos,
        "n_bins": profiles.n_bins,
        "percentiles": percentiles,
        "profiles": {},
    }

    if "sod_halo_bin_radius" in profiles.columns:
        result["radius_bins"] = np.median(
            profiles.get_column("sod_halo_bin_radius"), axis=0
        ).tolist()

    for col in columns:
        if col not in profiles.columns:
            logger.warning(f"Column '{col}' not found in profiles, skipping")
            continue
        pct_profiles = profiles.compute_percentile_profiles(col, percentiles)
        result["profiles"][col] = {k: v.tolist() for k, v in pct_profiles.items()}

    return result
