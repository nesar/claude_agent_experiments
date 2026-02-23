---
name: experiment
description: Start a new analysis experiment. Creates a timestamped experiment directory, saves the user prompt, and sets up the standard file structure for reproducible analysis.
allowed-tools: Bash(mkdir *), Bash(date *), Write, Read
argument-hint: [analysis goal]
---

# New Experiment Setup

Create a new experiment for the analysis goal described in: $ARGUMENTS

## Steps

1. Generate a timestamp with `date +%Y%m%d_%H%M%S`
2. Create the experiment directory at `/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/runs/experiment_<timestamp>/`
3. Save the user's analysis goal to `prompt.txt` in that directory
4. Plan the analysis:
   - Identify which MCP tools are needed (opencosmo-mcp for data, mcp-ke for analysis, kb-mcp for knowledge)
   - Determine if data needs to be fetched or is already local
   - Draft a step-by-step plan
5. Present the plan to the user before proceeding
6. All scripts, results, plots, and reports go into this experiment directory using absolute paths
7. Use OpenCosmo patterns for any data manipulation (never pandas)
8. When generating the report (via `/report`), the initial research question from `prompt.txt` will be displayed in a red query box in the Introduction section
