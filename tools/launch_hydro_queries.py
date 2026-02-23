#!/usr/bin/env python3
"""
Standalone script to launch two Globus Flows for HACC hydro catalog queries.

Uses the opencosmo-mcp package (installed in cosmodev env) to authenticate
with Globus and submit flow runs.

Queries:
1. Frontier-E hydro catalog, step=624 (z=0), sod_halo_M500c_low=1e13, limit=5000
2. SCIDAC_ALL hydro catalog, step=624 (z=0), limit=10000 (no mass filter)
"""

import os
import sys
import json
import logging

# Set environment variables BEFORE importing opencosmo_mcp (pydantic-settings reads them)
os.environ["OCMCP_DATA_IS_LOCAL"] = "false"
os.environ["OCMCP_DOWNLOAD_DIR"] = "/data/a/cpac/nramachandra/Projects/AmSC/tmp/OpenCosmo"
os.environ["OCMCP_WAIT_FOR_COMPLETION"] = "false"  # We just want run_ids

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

from opencosmo_mcp.settings import CLIENT_ID, get_settings
from opencosmo_mcp.globus import GlobusFlowClient
from opencosmo_mcp.adapter import FlowAdapter


def load_hydro_adapter():
    """Load the halo query hydro adapter from the adapter JSON file."""
    adapter_json_path = os.path.join(
        "/data/a/cpac/nramachandra/Projects/AmSC/opencosmo-mcp",
        "src", "adapters", "haloquery_hydro_flow.json"
    )
    with open(adapter_json_path, "r") as f:
        adapter_data = json.load(f)
    return FlowAdapter(adapter_data)


def main():
    # Load the hydro adapter
    adapter = load_hydro_adapter()
    flow_uuid = adapter.flow_uuid
    logger.info(f"Loaded adapter: {adapter.name} (flow_uuid={flow_uuid})")

    # Initialize the Globus client
    # The UserApp will use cached tokens from ~/.globus/app/{client_id}/
    settings = get_settings()
    app_name = settings.app_name  # "OpenCosmo Flow Launcher"
    logger.info(f"App name: {app_name}, Client ID: {CLIENT_ID}")

    client = GlobusFlowClient(
        client_id=CLIENT_ID,
        flow_ids=[flow_uuid],
        app_name=app_name,
    )

    # Trigger authentication (should use cached tokens)
    logger.info("Authenticating with Globus (using cached tokens)...")
    _ = client.user_app
    logger.info("Authentication successful!")

    # =========================================================================
    # Query 1: Frontier-E, step=624, M500c > 1e13, limit=5000
    # =========================================================================
    query1_params = {
        "simulation": "Frontier-E",
        "step": "624",
        "limit": 5000,
        "single": None,           # Not applicable for Frontier-E
        "include_profiles": False,
        "include_galaxies": False,
        "sod_halo_M500c_low": 1e13,
    }

    logger.info(f"Building flow input for Query 1 (Frontier-E)...")
    flow_input_1 = adapter.build_flow_input(**query1_params)
    # Wrap in "input" key as required by Globus Flows API
    wrapped_input_1 = {"input": flow_input_1}

    label_1 = "Halo Query: Frontier-E (s=624, M500c>1e13)"
    tags_1 = [f"adapter:{adapter.slug}", "opencosmo-mcp"]

    logger.info(f"Submitting Query 1: {label_1}")
    run_id_1 = client.run_flow(flow_uuid, wrapped_input_1, label=label_1, tags=tags_1)
    logger.info(f"Query 1 submitted! Run ID: {run_id_1}")

    # =========================================================================
    # Query 2: SCIDAC_ALL, step=624, no mass filter, limit=10000
    # =========================================================================
    query2_params = {
        "simulation": "SCIDAC_ALL",
        "step": "624",
        "limit": 10000,
        "single": False,
        "include_profiles": False,
        "include_galaxies": False,
    }

    logger.info(f"Building flow input for Query 2 (SCIDAC_ALL)...")
    flow_input_2 = adapter.build_flow_input(**query2_params)
    wrapped_input_2 = {"input": flow_input_2}

    label_2 = "Halo Query: SCIDAC_ALL (s=624, limit=10000)"
    tags_2 = [f"adapter:{adapter.slug}", "opencosmo-mcp"]

    logger.info(f"Submitting Query 2: {label_2}")
    run_id_2 = client.run_flow(flow_uuid, wrapped_input_2, label=label_2, tags=tags_2)
    logger.info(f"Query 2 submitted! Run ID: {run_id_2}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FLOW SUBMISSIONS COMPLETE")
    print("=" * 70)
    print(f"\nQuery 1 (Frontier-E, M500c>1e13, limit=5000):")
    print(f"  Run ID: {run_id_1}")
    print(f"\nQuery 2 (SCIDAC_ALL, no filter, limit=10000):")
    print(f"  Run ID: {run_id_2}")
    print(f"\nMonitor with:")
    print(f"  opencosmo-mcp get_flow_status tool")
    print(f"  or Globus web app: https://app.globus.org/runs")
    print(f"\nDownload dir: {settings.download_dir}")
    print("=" * 70)

    return run_id_1, run_id_2


if __name__ == "__main__":
    run_id_1, run_id_2 = main()
