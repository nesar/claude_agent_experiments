"""
Data loading, metadata extraction, and cross-simulation concatenation tools
for HACC hydro simulation catalogs (SCIDAC format).

Uses opencosmo.open() to load SimulationCollection objects.
"""
import logging
import numpy as np
import opencosmo as oc

logging.basicConfig(level=logging.INFO)

# Available local catalogs (catalog-only, no particle data)
CATALOG_DIR = "/data/a/cpac/nramachandra/Projects/AmSC/MACosmo"
CATALOGS = {
    "massive":  f"{CATALOG_DIR}/fb680d50-f287-4288-849f-7a756cf49727/filtered_haloproperties.hdf5",  # 9,080 halos, M > 3.5e13
    "top1k":    f"{CATALOG_DIR}/c1790dfe-809f-4e4e-9373-795215baa27a/filtered_haloproperties.hdf5",  # 64,000 halos (1000/sim)
    "top10k":   f"{CATALOG_DIR}/f0133c40-66f6-4a64-a2b0-0b59f92d49ed/filtered_haloproperties.hdf5",  # 640,000 halos (10000/sim)
}
DEFAULT_CATALOG = CATALOGS["massive"]


def load_catalog(path=DEFAULT_CATALOG, units=None):
    """
    Load an HDF5 halo catalog as an OpenCosmo SimulationCollection.

    Args:
        path: Path to the HDF5 file.
        units: Unit convention ('scalefree', 'comoving', 'physical', or None for default).

    Returns:
        oc.SimulationCollection: Dict-like object with sim_name -> Dataset mappings.
    """
    ds = oc.open(path)
    if units is not None:
        ds = ds.with_units(units)
    n_sims = len(ds)
    n_halos = sum(len(v) for v in ds.values())
    logging.info(f"Loaded {n_sims} simulations, {n_halos} total halos from {path}")
    return ds


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


def concat_sims(ds, columns=None):
    """
    Concatenate data across all simulations into numpy arrays.

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
    import os
    available = {}
    for name, path in CATALOGS.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            available[name] = path
            print(f"  {name:10s} -> {size_mb:7.1f} MB  {path}")
        else:
            print(f"  {name:10s} -> MISSING  {path}")
    return available


def load_with_particles(halo_properties_path, halo_particles_path, units="physical"):
    """
    Load halo catalog + particle data as an OpenCosmo StructureCollection.

    Requires two separate HDF5 files: one for halo properties, one for particles.
    Use with the `with` statement for proper resource cleanup.

    Args:
        halo_properties_path: Path to haloproperties HDF5 file.
        halo_particles_path: Path to haloparticles HDF5 file.
        units: Unit convention (default 'physical').

    Returns:
        StructureCollection that can be used with halos() iterator and visualize_halo().

    Example:
        with load_with_particles("haloproperties.hdf5", "haloparticles.hdf5") as data:
            halo = next(data.halos())
            halo_id = halo["halo_properties"]["unique_tag"]
    """
    data = oc.open(halo_properties_path, halo_particles_path)
    if units:
        data = data.with_units(units)
    logging.info(f"Loaded StructureCollection from {halo_properties_path} + {halo_particles_path}")
    return data
