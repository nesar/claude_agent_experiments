"""
OpenCosmo particle and structure collection tools.

Functions for working with halos + particle data (StructureCollections).
Requires two HDF5 files: halo properties and halo particles.
"""
import logging

import numpy as np
import opencosmo as oc

from oc_data import load_with_particles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iterate_halos(data, particle_types=None, n=None):
    """
    Iterate over halos yielding properties + particles.

    Args:
        data: StructureCollection (from oc.open or load_with_particles).
        particle_types: List of particle types to include.
                       Options: ["dm_particles", "gas_particles", "star_particles", "agn_particles"].
                       If None, defaults to ["dm_particles"].
        n: Max number of halos to iterate over. If None, iterates all.

    Yields:
        dict: Halo data with 'halo_properties' and particle type keys.

    Example:
        data = load_with_particles("props.hdf5", "parts.hdf5")
        for halo in iterate_halos(data, particle_types=["dm_particles", "star_particles"], n=10):
            mass = halo["halo_properties"]["fof_halo_mass"]
            dm = halo["dm_particles"]
    """
    if particle_types is None:
        particle_types = ["dm_particles"]

    if n is not None:
        data = data.take(n, at="start")

    for halo in data.halos(particle_types):
        yield halo


def get_halo_by_id(data, halo_id, id_column="fof_halo_tag", particle_types=None):
    """
    Get a single halo and its particles by ID.

    Args:
        data: StructureCollection.
        halo_id: The halo's unique identifier value.
        id_column: Column name containing halo IDs (default: 'fof_halo_tag').
        particle_types: List of particle types to include. Defaults to ["dm_particles"].

    Returns:
        dict: Halo data with 'halo_properties' and particle type keys.

    Raises:
        ValueError: If no halo found with the given ID.
    """
    if particle_types is None:
        particle_types = ["dm_particles"]

    filtered = data.filter(oc.col(id_column) == halo_id)

    for halo in filtered.halos(particle_types):
        return halo

    raise ValueError(f"No halo found with {id_column}={halo_id}")


def extract_halo_particles(data, halo_id, particle_type="dm_particles",
                           id_column="fof_halo_tag"):
    """
    Extract particles for a specific halo as a standalone dataset.

    Args:
        data: StructureCollection.
        halo_id: The halo's unique identifier value.
        particle_type: Type of particles to extract (default: 'dm_particles').
        id_column: Column name containing halo IDs.

    Returns:
        opencosmo Dataset containing just the particles for this halo.

    Raises:
        ValueError: If no halo or no particles of the requested type found.
    """
    filtered = data.filter(oc.col(id_column) == halo_id)

    for halo in filtered.halos([particle_type]):
        if particle_type in halo:
            return halo[particle_type]

    raise ValueError(f"No {particle_type} found for halo {id_column}={halo_id}")


def get_halo_info(data, halo_id, id_column="fof_halo_tag",
                  particle_types=None, return_columns=None):
    """
    Get summary information about a halo and its particle counts.

    Args:
        data: StructureCollection.
        halo_id: The halo's unique identifier value.
        id_column: Column name containing halo IDs.
        particle_types: List of particle types to check. Defaults to all available.
        return_columns: Halo property columns to include.

    Returns:
        dict with 'halo_properties' (dict of values) and 'particles' (dict of counts).
    """
    if particle_types is None:
        particle_types = ["dm_particles", "gas_particles", "star_particles", "agn_particles"]

    if return_columns is None:
        return_columns = [
            id_column, "fof_halo_mass", "sod_halo_mass",
            "sod_halo_radius", "sod_halo_cdelta",
        ]

    halo = get_halo_by_id(data, halo_id, id_column, particle_types)

    props = {}
    halo_props = halo["halo_properties"]
    for col in return_columns:
        if col in halo_props:
            val = halo_props[col]
            props[col] = float(val) if isinstance(val, (np.integer, np.floating)) else val

    particle_info = {}
    for ptype in particle_types:
        if ptype in halo:
            particles = halo[ptype]
            particle_info[ptype] = {
                "count": len(particles),
                "columns": list(particles.columns) if hasattr(particles, 'columns') else [],
            }

    return {"halo_properties": props, "particles": particle_info}


def visualize_halo(data, halo_id, output_path, projection_axis="z", width=None):
    """
    Create a multi-panel particle projection visualization for a single halo.

    Uses opencosmo.analysis.visualize_halo to generate the figure.

    Args:
        data: StructureCollection with particle data.
        halo_id: The halo's unique identifier (fof_halo_tag or unique_tag).
        output_path: Path to save the output image (PNG).
        projection_axis: Projection direction ('x', 'y', or 'z').
        width: Panel width in R200 units (default: auto).

    Returns:
        matplotlib.figure.Figure
    """
    from opencosmo.analysis import visualize_halo as oc_visualize_halo

    kwargs = {"halo_id": halo_id, "data": data, "projection_axis": projection_axis}
    if width is not None:
        kwargs["width"] = width

    fig = oc_visualize_halo(**kwargs)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved halo visualization to {output_path}")

    return fig
