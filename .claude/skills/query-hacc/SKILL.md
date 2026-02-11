---
name: query-hacc
description: Query HACC simulation data using the opencosmo-mcp server. Use when the user needs halo catalogs, particle data, galaxy catalogs, or lightcone data from HACC simulations.
argument-hint: [query description]
---

# Query HACC Simulation Data

Query goal: $ARGUMENTS

## Available Query Types (via opencosmo-mcp)

- **Halo catalogs (gravity-only):** `launch_halo_catalog_query_gravity` -- Frontier-E GO simulation
- **Halo catalogs (hydro):** `launch_halo_catalog_query_hydro` -- hydrodynamical simulation
- **Halo particles (gravity):** `launch_halo_particles_query_gravity` -- particle data around halos
- **Halo particles (hydro):** `launch_halo_particles_query_hydro` -- gas, DM, star, AGN particles
- **Halo lightcone (gravity):** `launch_halo_catalog_query_lc_gravity` -- lightcone halo data
- **Galaxy catalogs:** `launch_galaxy_query` -- galaxy properties
- **Synthetic galaxies:** `launch_synthetic_galaxy_query` -- synthetic galaxy data
- **Maps:** `launch_map_query` -- projected maps

## Workflow

1. Determine which query type matches the user's needs
2. Call the appropriate `launch_*` tool with parameters (simulation, step, filters, limit)
3. The flow runs on Globus; since `OCMCP_WAIT_FOR_COMPLETION=true`, it will wait and return results
4. If it times out, use `get_flow_status` with the run_id to check progress
5. Once complete, load the result file with OpenCosmo: `oc.open("/path/to/result.hdf5")`
6. Access data with `np.array(dataset.data["column_name"])` -- never `.to_pandas()`

## Monitoring

- `get_flow_status(run_id)` -- check progress of a running flow
- `list_recent_flows(limit)` -- see recent flow runs
- `cancel_flow(run_id)` -- cancel a running flow
