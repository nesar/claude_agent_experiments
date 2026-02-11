# How to Use Skills in Claude Code

## What Are Skills?

Skills are reusable workflow templates you invoke with `/skill-name` in Claude Code. Each skill is a markdown file (`SKILL.md`) containing instructions that Claude follows when you invoke it. They live in `.claude/skills/<skill-name>/SKILL.md`.

Skills are **not** the same as MCP tools:

| | MCP Tools | Skills |
|---|---|---|
| **What they are** | External server capabilities (query data, compute P(k), search KB) | Workflow templates with step-by-step instructions |
| **How invoked** | Claude calls them automatically during conversation | You type `/skill-name [args]` |
| **Where defined** | In MCP server source code (immutable) | In `.claude/skills/` (editable) |
| **Role** | Provide atomic operations | Orchestrate those operations into structured workflows |

Think of MCP tools as **verbs** (compute, search, query) and skills as **recipes** that combine those verbs into a coherent workflow.

## Available Skills in This Project

| Command | Purpose | Example |
|---------|---------|---------|
| `/experiment [goal]` | Start a new analysis session | `/experiment Find bullet-like clusters in hydro sim` |
| `/query-hacc [description]` | Query HACC simulation data via Globus | `/query-hacc massive halos with particle data` |
| `/power-spectrum [description]` | P(k) analysis, model comparison, MCMC | `/power-spectrum Compare LCDM with massive neutrino models` |
| `/literature [topic]` | Search arXiv + knowledge base for references | `/literature galaxy cluster merger signatures` |
| `/report [experiment dir]` | Generate LaTeX report from results | `/report runs/experiment_20260208_143022` |

## How to Use Them

### Basic Usage

Type `/` in Claude Code to see the list of available skills, then select one or type the full name:

```
> /experiment Analyze gas fractions in the 50 most massive clusters
```

Claude receives the skill instructions with your text substituted as the argument, then follows the workflow.

### Chaining Skills in a Session

Skills can be used sequentially in a single conversation:

1. `/experiment [goal]` -- set up the experiment directory and plan
2. `/query-hacc [what data]` -- fetch simulation data
3. `/literature [topic]` -- gather references and methodology
4. (do the analysis interactively or via scripts)
5. `/report [experiment dir]` -- compile everything into a PDF

### What Happens When You Invoke a Skill

1. Claude reads the `SKILL.md` file for that skill
2. Your arguments replace `$ARGUMENTS` (or `$0`, `$1`, etc.) in the instructions
3. Claude follows the instructions, using MCP tools and Claude Code tools as needed
4. You interact normally -- Claude may ask clarifying questions or present plans

## Anatomy of a Skill File

Each skill lives in `.claude/skills/<name>/SKILL.md` with this format:

```yaml
---
name: skill-name
description: When and why to use this skill
allowed-tools: Read, Write, Bash(git *)
argument-hint: [what arguments to provide]
---

# Skill Title

Instructions that Claude follows when this skill is invoked.

Your arguments are available as: $ARGUMENTS
First argument: $0
Second argument: $1
```

### Key Frontmatter Fields

| Field | What it does |
|-------|-------------|
| `name` | Skill name (used as `/name`) |
| `description` | When to use it; Claude also reads this to decide if it should auto-invoke |
| `allowed-tools` | Tools Claude can use without asking permission during this skill |
| `argument-hint` | Hint shown in autocomplete (e.g., `[topic]`, `[file] [format]`) |
| `disable-model-invocation` | Set `true` to prevent Claude from auto-triggering it |
| `context: fork` | Run in an isolated subagent (won't pollute main conversation context) |

## Creating a New Skill

1. Create a directory: `.claude/skills/my-skill/`
2. Write `SKILL.md` with frontmatter + instructions
3. Restart Claude Code (or it picks up changes on next `/` invocation)
4. Invoke with `/my-skill [arguments]`

### Example: A Custom Analysis Skill

```yaml
---
name: scaling-relations
description: Compute and plot scaling relations between halo properties
argument-hint: [X property] [Y property]
---

# Scaling Relations Analysis

Compute the scaling relation between $0 and $1.

1. Query halo catalog data using opencosmo-mcp if not already available
2. Load data with OpenCosmo: `oc.open("path/to/catalog.hdf5")`
3. Extract columns: `np.array(dataset.data["$0"])` and `np.array(dataset.data["$1"])`
4. Fit a power-law relation in log-log space
5. Plot with matplotlib, including best-fit line and scatter
6. Save to the current experiment directory
```

Invoke with: `/scaling-relations sod_halo_M500c sod_halo_T500c`

## Tips

- **Type `/` to browse** all available skills interactively
- **Skills complement MCP tools** -- use skills for workflows, MCP tools for data access
- **Skills are just markdown** -- easy to read, edit, and version control
- **Arguments are flexible** -- anything after the skill name becomes `$ARGUMENTS`
- **Skills can include shell commands** -- prefix with `!` to run before Claude sees the content (e.g., `!`git status``)
- **Add supporting files** alongside `SKILL.md` in the skill directory if needed (reference docs, example scripts)

## File Locations

```
.claude/skills/
├── experiment/SKILL.md       # /experiment
├── query-hacc/SKILL.md       # /query-hacc
├── power-spectrum/SKILL.md   # /power-spectrum
├── literature/SKILL.md       # /literature
└── report/SKILL.md           # /report
```

Global skills (available in all projects) go in `~/.claude/skills/`.
Project skills (this project only) go in `.claude/skills/`.
