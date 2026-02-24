# Claude Agent Experiments -- Multi-Agent System for Cosmological Analysis

## Critical Rules

1. **NEVER MODIFY MCP SERVER CODE**: The three MCP server repos are IMMUTABLE. Never edit files in `opencosmo-mcp/`, `mcp-ke/`, or `kb-mcp/`.
2. **ABSOLUTE PATHS ONLY**: Always use `/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/runs/experiment_*/` -- never relative paths.
3. **NO PANDAS ON OPENCOSMO DATA**: Never use `.to_pandas()` on OpenCosmo datasets. Use `np.array(dataset.data["column"])` instead.
4. **NO SYNTHETIC DATA**: All results must come from real datasets via MCP tools or local OpenCosmo files. Never generate fake/placeholder data.
5. **TOOLS DIR IS FOR LOCAL UTILITIES ONLY**: The `tools/` directory is for any local helper scripts. Do not duplicate MCP server functionality here.

## 1. Project Context

- **HACC Simulation Analysis**: Cosmological simulation data from the Hardware/Hybrid Accelerated Cosmology Code (HACC), including gravity-only and hydrodynamical runs.
- **Data Types**:
  - **Halo/cluster catalogs**: HDF5 files with ~80 attributes per halo (mass, temperature, luminosity, gas fractions, etc.) via OpenCosmo
  - **Particle files**: Dark matter, gas, AGN, star particles within regions surrounding clusters
  - **Power spectra**: Matter power spectrum P(k) from eBOSS and theoretical models
  - **Knowledge base**: Documents, papers, and references searchable via semantic search
- **Objective**: Automated, reproducible cosmological analyses -- cluster selection, derived metrics, power spectrum fitting, MCMC parameter estimation, literature integration, and publication-quality outputs.

## 2. MCP Servers

All three servers use stdio transport and are configured in `.mcp.json`.

### 2.1 opencosmo-mcp (Simulation Data via Globus Flows)
- **Location:** `/data/a/cpac/nramachandra/Projects/AmSC/opencosmo-mcp`
- **Purpose:** Query HACC simulation data via Globus Flows -- launches remote computations and retrieves results
- **Launch tools:** `launch_halo_catalog_query_gravity`, `launch_halo_catalog_query_hydro`, `launch_halo_particles_query_gravity`, `launch_halo_particles_query_hydro`, `launch_halo_catalog_query_lc_gravity`, `launch_galaxy_query`, `launch_synthetic_galaxy_query`, `launch_map_query`
- **Management tools:** `get_flow_status`, `cancel_flow`, `list_recent_flows`, `cleanup_downloads`
- **Auth:** Globus OAuth (interactive browser login on first run)
- **Config:** `OCMCP_WAIT_FOR_COMPLETION=true` (waits for flow completion), `OCMCP_DATA_IS_LOCAL=true`
- **Workflow:** Launch query -> flow runs on Globus -> poll status -> retrieve result path -> load with OpenCosmo

### 2.2 mcp-ke (Cosmology Analysis Tools)
- **Location:** `/data/a/cpac/nramachandra/Projects/AmSC/mcp-ke`
- **Purpose:** Power spectrum analysis, MCMC parameter estimation, model comparison, arXiv search
- **Data tools:** `load_observational_data`, `create_theory_k_grid`
- **Model tools:** `get_lcdm_params`, `get_nu_mass_params`, `get_wcdm_params`
- **Compute tools:** `compute_power_spectrum`, `compute_all_models`, `compute_suppression_ratios`
- **Viz tools:** `plot_power_spectra`, `plot_suppression_ratios`
- **MCMC tools:** `run_mcmc_cosmology`, `create_mcmc_corner_plot`, `create_mcmc_trace_plot`, `analyze_mcmc_samples`, `compute_best_fit_power_spectrum`
- **Utility tools:** `save_array`, `load_array`, `save_dict`, `load_dict`, `list_agent_files`
- **Agent tools:** `arxiv_agent` (literature search + paper analysis)
- **Config:** `GOOGLE_API_KEY` env var for agent tools; core analysis works without it

### 2.3 kb-mcp (Knowledge Base & Semantic Search)
- **Location:** `/data/a/cpac/nramachandra/Projects/AmSC/kb-mcp`
- **Purpose:** Document storage, semantic search, retrieval for domain knowledge
- **Tools:** `kb_search` (hybrid semantic + fulltext), `kb_get` (retrieve full doc), `kb_get_image`
- **Resources:** `kb://sources`, `kb://document/{doc_id}`
- **Config:** `DISABLE_AUTH=true` for local dev; needs `OPENAI_API_KEY` for embeddings

## 3. OpenCosmo Data Patterns

**Reference docs available at:** `docs/opencosmo_docs.jsonl` and `docs/catalog_attributes.txt`
**Online docs:** https://opencosmo.readthedocs.io/en/latest/index.html

### Data Loading
```python
import opencosmo as oc
import numpy as np

# Load halo catalog
dataset = oc.open("haloproperties.hdf5")

# Access columns -- ALWAYS use these patterns
masses = np.array(dataset.data["sod_halo_mass"])
temps = np.array(dataset.data["sod_halo_T500c"])

# NEVER do this:
# df = dataset.to_pandas()  # FORBIDDEN
```

### Filtering, Selection, Sorting
```python
# Filter
filtered = dataset.filter(oc.col("fof_halo_mass") > 1e14)

# Select columns
selected = dataset.select(["fof_halo_mass", "sod_halo_cdelta"])

# Sort (descending) and take top N
top100 = dataset.sort_by("fof_halo_mass", invert=True).take(100, at="start")

# Add computed columns
dataset = dataset.with_new_columns(
    gas_fraction=oc.col("sod_halo_MGas500c") / oc.col("sod_halo_M500c")
)

# Chain operations
result = (dataset
    .filter(oc.col("fof_halo_mass") > 1e13)
    .select(["fof_halo_mass", "sod_halo_cdelta"])
    .sort_by("fof_halo_mass", invert=True)
    .take(1000, at="start"))
```

### Particle Data (Structure Collections)
```python
# Load halo properties + particles together
data = oc.open("haloproperties.hdf5", "haloparticles.hdf5")

structures = data.filter(oc.col("fof_halo_mass") > 1e14).take(10, at="random")
for halo in structures.halos(["dm_particles", "star_particles"]):
    halo_mass = halo["halo_properties"]["fof_halo_mass"]
    dm = halo["dm_particles"]
    stars = halo["star_particles"]
```

### Unit Conventions
- `dataset.with_units("scalefree")` -- comoving Mpc/h, Msun/h
- `dataset.with_units("comoving")` -- factors of h absorbed (default)
- `dataset.with_units("physical")` -- physical coordinates

### Key Catalog Attributes
See `docs/catalog_attributes.txt` for the full list (~80 columns). Key ones:
- **Mass:** `fof_halo_mass`, `sod_halo_mass`, `sod_halo_M500c`, `sod_halo_M200m`, `sod_halo_MVir`
- **Temperature:** `sod_halo_T500c`, `sod_halo_T500cBolo`, `sod_halo_T500cBoloEx`
- **Gas:** `sod_halo_MGas500c`, `sod_halo_MGasHot500c`, `sod_halo_GasFracShell2500c`
- **X-ray:** `sod_halo_L500cBolo`, `sod_halo_L500cRosat`, `sod_halo_L500cErositaHi`
- **Structure:** `sod_halo_cdelta`, `sod_halo_c_acc_mass`, `sod_halo_core_cusp`, `sod_halo_core_entropy`
- **SZ:** `sod_halo_Y500c`, `sod_halo_Y5R500c`
- **Stellar:** `sod_halo_MStar500c`, `sod_halo_sfr`
- **AGN:** `sod_halo_mmagn_mass`, `sod_halo_mass_agn`
- **IDs:** `fof_halo_tag`, `unique_tag`

## 4. Workflow

### 4.1 Planning
- Restate user goals, draft a high-level plan, identify which MCP tools are needed
- Determine if data needs to be fetched (opencosmo-mcp) or is already local
- Check kb-mcp for relevant domain knowledge and papers

### 4.2 Research
- Use `arxiv_agent` (mcp-ke) for literature search on methodologies
- Use `kb_search` (kb-mcp) for stored domain knowledge
- Use web search for supplementary references
- Examine dataset metadata and available attributes before analysis
- Include citations and methodology references in results

### 4.3 Analysis
- **FIRST**: Use MCP server tools for their intended purposes
- **THEN**: Write custom analysis scripts using OpenCosmo patterns for anything beyond MCP tool capabilities
- Write standalone scripts in experiment directories (never modify MCP server code)
- Use numpy arrays for numerical operations: `np.array(dataset.data["column"])`

### 4.4 Visualization
- Publication-quality plots with matplotlib
- For particle projections, use OpenCosmo's built-in visualization or yt
- Save all plots as individual PNG files

### 4.5 Archiving (Automatic)
Every analysis session creates a timestamped experiment directory:
```
runs/experiment_YYYYMMDD_HHMMSS/
├── prompt.txt              # Original user request
├── code.py                 # Executable analysis script(s)
├── results.csv/json        # Output data
├── log.txt                 # Analysis log
├── methodology.md          # Web/literature research findings
├── *.png                   # All visualizations
├── report.tex              # LaTeX manuscript (if applicable)
├── report.pdf              # Compiled PDF (if applicable)
└── report.bib              # Bibliography
```

## 5. Environment

- **Python:** 3.12 via conda env `cosmodev` (`/home/nramachandra/anaconda3/envs/cosmodev/bin/python3`)
- **Package management:** `pip` (no `uv` available)
- **MCP SDK:** v1.22.0
- **Key packages:** opencosmo, classy (CLASS), emcee, getdist, matplotlib, numpy, scipy
- **API keys:** `.env` file in project root (GOOGLE_API_KEY, OPENAI_API_KEY)
- **Project root:** `/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments`

## 6. Skills (Slash Commands)

Skills are reusable workflow templates invoked with `/skill-name`. They provide structured
instructions that combine MCP tools with analysis patterns. Defined in `.claude/skills/`.

| Command | Purpose |
|---------|---------|
| `/experiment [goal]` | Full automated analysis -- creates timestamped dir, plans & executes, generates report |
| `/research [question]` | Interactive step-by-step research -- human-in-the-loop, phases tracked, asks "what next?" |
| `/query-hacc [description]` | Query HACC simulation data via opencosmo-mcp Globus Flows |
| `/power-spectrum [description]` | Run P(k) analysis, model comparison, or MCMC fitting via mcp-ke |
| `/literature [topic]` | Search arXiv (mcp-ke) and knowledge base (kb-mcp) for references |
| `/report [experiment dir]` | Generate LaTeX report -- red query boxes, multi-phase support for `/research` sessions |
| `/demo-video [experiment dir or description]` | Generate animated MP4 demo video from a completed experiment -- invoke AFTER analysis is complete |

**How skills work:** Each skill is a `SKILL.md` file in `.claude/skills/<name>/` containing
a YAML frontmatter (metadata) and markdown body (instructions). When you type `/experiment
find bullet-like clusters`, Claude receives the skill instructions with your arguments
substituted in, then follows the workflow using the available MCP tools and Claude Code tools.
Skills don't replace MCP tools -- they orchestrate them.

## 7. Directory Structure

```
./
├── CLAUDE.md                    # This file
├── .mcp.json                    # MCP server configuration (3 servers)
├── .env                         # API keys (GOOGLE_API_KEY, OPENAI_API_KEY)
├── .claude/
│   ├── settings.local.json      # Claude Code local settings
│   └── skills/                  # Skill definitions (slash commands)
│       ├── experiment/SKILL.md
│       ├── query-hacc/SKILL.md
│       ├── power-spectrum/SKILL.md
│       ├── literature/SKILL.md
│       └── report/SKILL.md
├── docs/
│   ├── catalog_attributes.txt   # Full list of ~80 halo catalog columns with descriptions
│   ├── opencosmo_docs.jsonl     # Scraped OpenCosmo documentation (26 pages)
│   └── scrape_opencosmo_json.py # Script that generated the docs jsonl
├── tools/                       # Local helper scripts (if needed beyond MCP tools)
└── runs/                        # Auto-saved experiment artifacts
    └── experiment_<timestamp>/
```

## 7. Script Template

```python
#!/usr/bin/env python3
"""Analysis script for experiment_YYYYMMDD_HHMMSS"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import opencosmo as oc

# Absolute paths
PROJECT_ROOT = "/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "runs", "experiment_YYYYMMDD_HHMMSS")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

def main():
    # Load data (from local file or MCP-retrieved path)
    dataset = oc.open("/path/to/catalog.hdf5")

    # Access data correctly
    masses = np.array(dataset.data["sod_halo_mass"])  # RIGHT
    # df = dataset.to_pandas()  # WRONG - NEVER DO THIS

    # Filter, analyze, visualize
    massive = dataset.filter(oc.col("fof_halo_mass") > 1e14)

    # Save results
    fig, ax = plt.subplots()
    ax.hist(np.log10(masses), bins=50)
    fig.savefig(os.path.join(EXPERIMENT_DIR, "mass_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
```

## 8. References

- **OpenCosmo Documentation:** https://opencosmo.readthedocs.io/en/latest/index.html
  - Column reference: https://opencosmo.readthedocs.io/en/latest/cols.html
  - API reference: https://opencosmo.readthedocs.io/en/latest/api.html
- **Local docs:** `docs/opencosmo_docs.jsonl` (26 pages of scraped documentation)
- **Catalog attributes:** `docs/catalog_attributes.txt` (full column descriptions)
