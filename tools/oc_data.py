"""
Pure OpenCosmo data loading, metadata extraction, and cross-simulation concatenation.

Uses opencosmo.open() to load SimulationCollection objects.
All functions take/return native opencosmo objects (no session management, no pandas).
"""
import logging
import os
import re

import h5py
import numpy as np
import opencosmo as oc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available local catalogs (catalog-only, no particle data)
CATALOG_DIR = "/data/a/cpac/nramachandra/Projects/AmSC/MACosmo"
CATALOGS = {
    "massive":  f"{CATALOG_DIR}/fb680d50-f287-4288-849f-7a756cf49727/filtered_haloproperties.hdf5",
    "top1k":    f"{CATALOG_DIR}/c1790dfe-809f-4e4e-9373-795215baa27a/filtered_haloproperties.hdf5",
    "top10k":   f"{CATALOG_DIR}/f0133c40-66f6-4a64-a2b0-0b59f92d49ed/filtered_haloproperties.hdf5",
}
DEFAULT_CATALOG = CATALOGS["massive"]


def load_catalog(path=DEFAULT_CATALOG, units=None):
    """
    Load an HDF5 halo catalog as an OpenCosmo SimulationCollection.

    Args:
        path: Path to the HDF5 file.
        units: Unit convention ('scalefree', 'comoving', 'physical', or None for default).

    Returns:
        SimulationCollection: Dict-like object with sim_name -> Dataset mappings.
    """
    ds = oc.open(path)
    if units is not None:
        ds = ds.with_units(units)
    n_sims = len(ds)
    n_halos = sum(len(v) for v in ds.values())
    logger.info(f"Loaded {n_sims} simulations, {n_halos} total halos from {path}")
    return ds


def load_with_particles(halo_properties_path, halo_particles_path, units="physical"):
    """
    Load halo catalog + particle data as an OpenCosmo StructureCollection.

    Requires two separate HDF5 files: one for halo properties, one for particles.

    Args:
        halo_properties_path: Path to haloproperties HDF5 file.
        halo_particles_path: Path to haloparticles HDF5 file.
        units: Unit convention (default 'physical').

    Returns:
        StructureCollection that can be used with halos() iterator.
    """
    data = oc.open(halo_properties_path, halo_particles_path)
    if units:
        data = data.with_units(units)
    logger.info(f"Loaded StructureCollection from {halo_properties_path} + {halo_particles_path}")
    return data


def describe_catalog(ds):
    """
    Print summary metadata for a SimulationCollection.

    Args:
        ds: SimulationCollection from load_catalog().

    Returns:
        dict: Summary metadata.
    """
    sim_names = list(ds.keys())
    first_ds = ds[sim_names[0]]
    columns = sorted(first_ds.columns)
    counts = {name: len(ds[name]) for name in sim_names}
    total = sum(counts.values())

    cosmo = ds.cosmology
    first_cosmo = cosmo[sim_names[0]]
    redshift = ds.redshift[sim_names[0]]

    info = {
        "n_simulations": len(sim_names),
        "n_halos_total": total,
        "n_columns": len(columns),
        "columns": columns,
        "halos_per_sim": counts,
        "redshift": redshift,
        "cosmology": {
            "H0": float(first_cosmo.H0.value),
            "Om0": float(first_cosmo.Om0),
            "Ob0": float(first_cosmo.Ob0),
        },
    }
    print(f"Simulations: {info['n_simulations']}")
    print(f"Total halos: {info['n_halos_total']}")
    print(f"Columns: {info['n_columns']}")
    print(f"Redshift: {info['redshift']}")
    print(f"Cosmology: H0={info['cosmology']['H0']:.2f}, Om0={info['cosmology']['Om0']:.4f}, Ob0={info['cosmology']['Ob0']:.5f}")
    return info


def get_columns(ds):
    """
    List all available columns in the catalog.

    Args:
        ds: SimulationCollection or Dataset.

    Returns:
        list: Sorted column names.
    """
    if isinstance(ds, dict):
        first = next(iter(ds.values()))
        return sorted(first.columns)
    return sorted(ds.columns)


def get_cosmology(ds):
    """
    Extract cosmology parameters (same across all SCIDAC sims).

    Args:
        ds: SimulationCollection.

    Returns:
        astropy.cosmology.FlatLambdaCDM: The cosmology object.
    """
    first_key = next(iter(ds.keys()))
    return ds.cosmology[first_key]


def list_catalogs():
    """
    Print available local catalogs and their sizes.

    Returns:
        dict: {name: path} for existing catalogs.
    """
    available = {}
    for name, path in CATALOGS.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            available[name] = path
            print(f"  {name:10s} -> {size_mb:7.1f} MB  {path}")
        else:
            print(f"  {name:10s} -> MISSING  {path}")
    return available


def concat_sims(ds, columns=None):
    """
    Concatenate data across all simulations into numpy arrays.

    This is the bridge from OpenCosmo SimulationCollection to numpy.
    Call this when you need arrays for analysis_tools or viz_tools.

    Args:
        ds: SimulationCollection (possibly filtered/transformed).
        columns: List of column names to extract. If None, extracts all.

    Returns:
        dict: {column_name: np.ndarray} with data concatenated across all sims.
    """
    first = next(iter(ds.values()))
    if columns is None:
        columns = list(first.columns)

    result = {col: [] for col in columns}
    for sim_name, sim_ds in ds.items():
        for col in columns:
            result[col].append(np.array(sim_ds.data[col]))

    return {col: np.concatenate(arrs) for col, arrs in result.items()}


def set_units(ds, convention):
    """
    Change the unit convention of a SimulationCollection or Dataset.

    Args:
        ds: SimulationCollection or Dataset.
        convention: Unit convention ('scalefree', 'comoving', 'physical').

    Returns:
        SimulationCollection/Dataset with new unit convention.
    """
    return ds.with_units(convention)


# ---------------------------------------------------------------------------
# File inspection (adapted from MACosmo extract_simulation_metadata)
# ---------------------------------------------------------------------------

def _safe_float(val):
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_hdf5_attrs(attrs):
    """Convert HDF5 attributes to a JSON-serializable dict."""
    result = {}
    for key in attrs:
        val = attrs[key]
        if isinstance(val, bytes):
            result[key] = val.decode('utf-8', errors='replace')
        elif isinstance(val, np.ndarray):
            result[key] = val.tolist()
        elif isinstance(val, np.integer):
            result[key] = int(val)
        elif isinstance(val, np.floating):
            result[key] = float(val)
        elif isinstance(val, (int, float, str, bool)):
            result[key] = val
        else:
            result[key] = str(val)
    return result


def _is_scidac_format(f):
    """Check if file uses SCIDAC partitioning (SCIDAC_000, SCIDAC_001, etc.)."""
    keys = list(f.keys())
    if not keys:
        return False
    scidac_pattern = re.compile(r'^SCIDAC_\d{3}$')
    return all(scidac_pattern.match(k) for k in keys)


def extract_simulation_metadata(path):
    """
    Extract metadata from an HDF5 file without loading it into memory.

    Inspects the file to determine its format (opencosmo, SCIDAC, raw HDF5)
    and extracts cosmological parameters, simulation info, column listings,
    and file structure.

    Args:
        path: Path to the HDF5 file.

    Returns:
        dict: Comprehensive file metadata including format, cosmology, columns, etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    result = {
        "path": path,
        "file_format": "unknown",
        "file_size_mb": round(file_size_mb, 2),
        "cosmology": None,
        "header": None,
        "columns": [],
        "column_count": 0,
        "structure": {
            "top_level_keys": [],
            "datasets_available": None,
            "n_scidac_blocks": None,
        },
        "hdf5_attributes": None,
        "row_count": None,
    }

    # Read HDF5 structure
    with h5py.File(path, 'r') as f:
        result["structure"]["top_level_keys"] = list(f.keys())

        root_attrs = _extract_hdf5_attrs(f.attrs)
        if root_attrs:
            result["hdf5_attributes"] = root_attrs

        if _is_scidac_format(f):
            result["file_format"] = "scidac"
            block_names = sorted([k for k in f.keys() if k.startswith('SCIDAC_')])
            result["structure"]["n_scidac_blocks"] = len(block_names)

            if block_names:
                first_block = f[block_names[0]]
                available_datasets = list(first_block.keys())
                result["structure"]["datasets_available"] = available_datasets

                # Two SCIDAC layouts:
                # 1. SCIDAC_XXX/halo_properties/data/{columns} (MACosmo style)
                # 2. SCIDAC_XXX/data/{columns} (opencosmo filtered style)
                data_group = None
                data_path = None

                for dataset_type in ["halo_properties", "halo_profiles"]:
                    if dataset_type in available_datasets:
                        ds_group = first_block[dataset_type]
                        if "data" in ds_group:
                            data_group = ds_group["data"]
                            data_path = f"{dataset_type}/data"
                            break

                if data_group is None and "data" in available_datasets:
                    data_group = first_block["data"]
                    data_path = "data"

                if data_group is not None:
                    columns = [k for k in data_group.keys()
                               if isinstance(data_group[k], h5py.Dataset)]
                    result["columns"] = columns
                    result["column_count"] = len(columns)
                    if columns:
                        first_col = columns[0]
                        total_rows = sum(
                            f[f'{bn}/{data_path}/{first_col}'].shape[0]
                            for bn in block_names
                        )
                        result["row_count"] = total_rows

            # Try to extract cosmology from attributes
            cosmo_keys = {"H0": ["H0", "hubble"], "Om0": ["Om0", "Omega_m"], "Ob0": ["Ob0", "Omega_b"]}
            all_attrs = _extract_hdf5_attrs(f.attrs)
            for key in list(f.keys())[:5]:
                if hasattr(f[key], 'attrs'):
                    all_attrs.update(_extract_hdf5_attrs(f[key].attrs))
            cosmo = {}
            for param, aliases in cosmo_keys.items():
                for alias in aliases:
                    if alias in all_attrs:
                        cosmo[param] = _safe_float(all_attrs[alias])
                        break
            if cosmo:
                result["cosmology"] = cosmo

            return result

    # Try opencosmo for non-SCIDAC files
    try:
        ds = oc.open(path)
        result["file_format"] = "opencosmo"

        if hasattr(ds, "cosmology") and ds.cosmology is not None:
            cosmo = ds.cosmology
            result["cosmology"] = {
                "H0": float(cosmo.H0.value) if hasattr(cosmo.H0, "value") else float(cosmo.H0),
                "Om0": float(cosmo.Om0),
                "Ob0": float(cosmo.Ob0) if hasattr(cosmo, "Ob0") else None,
            }

        if hasattr(ds, "columns"):
            result["columns"] = list(ds.columns)
            result["column_count"] = len(result["columns"])

        try:
            result["row_count"] = len(ds)
        except Exception:
            pass

        return result
    except Exception:
        pass

    # Fallback: raw HDF5 inspection
    result["file_format"] = "raw_hdf5"
    with h5py.File(path, 'r') as f:
        columns = []
        row_count = None
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset) and item.ndim == 1:
                columns.append(key)
                if row_count is None:
                    row_count = item.shape[0]
            elif isinstance(item, h5py.Group) and "data" in item:
                data_group = item["data"]
                for col_name in data_group.keys():
                    if isinstance(data_group[col_name], h5py.Dataset):
                        ds_item = data_group[col_name]
                        if ds_item.ndim == 1:
                            columns.append(col_name)
                            if row_count is None:
                                row_count = ds_item.shape[0]
        result["columns"] = columns
        result["column_count"] = len(columns)
        result["row_count"] = row_count

    return result
