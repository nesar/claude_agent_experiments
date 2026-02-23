---
name: research
description: Interactive, step-by-step research mode with human-in-the-loop. After each analysis step, pauses for user direction. Tracks phases for multi-stage reports. Use for exploratory analyses where the next step depends on prior results.
allowed-tools: Bash(mkdir *), Bash(date *), Bash(pdflatex *), Bash(bibtex *), Bash(cd *), Write, Read, Glob, AskUserQuestion
argument-hint: [initial research question]
---

# Interactive Research Session

Initial question: $ARGUMENTS

## Overview

This skill runs an **interactive research session** -- a back-and-forth workflow where analysis
proceeds in phases, with the user guiding direction after each step. Unlike `/experiment` which
plans and executes a full analysis upfront, `/research` is designed for exploratory work where
the next question depends on what you just found.

**Key differences from `/experiment`:**
- Human-in-the-loop: asks "what next?" after each phase
- Saves executable code for every phase (reproducibility)
- Report is written incrementally -- updated after each phase, not just at the end

## Setup

1. Generate a timestamp with `date +%Y%m%d_%H%M%S`
2. Create the experiment directory at `/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/runs/research_<timestamp>/`
3. Save the initial question to `prompt.txt`
4. Create a `phases.json` file to track the session's phases:
   ```json
   {
     "mode": "research",
     "phases": [
       {
         "phase": 1,
         "query": "<the initial question>",
         "status": "in_progress",
         "artifacts": []
       }
     ]
   }
   ```

## Phase Workflow

For each phase (starting with Phase 1 = the initial question):

1. **Understand the query** -- restate it briefly, identify what's needed
2. **Execute the analysis** -- use whatever tools are appropriate:
   - opencosmo-mcp for HACC data queries
   - mcp-ke for power spectrum, MCMC, arXiv search
   - kb-mcp for knowledge base lookups
   - Local OpenCosmo for data manipulation
   - Custom scripts for specialized analysis
3. **Save code** -- write the analysis code for this phase to a standalone script:
   `phase<N>_code.py`. This script must be **self-contained and executable**:
   - Include all imports, absolute paths, data loading, analysis, and plot saving
   - Follow the same patterns as `/experiment` scripts (see CLAUDE.md Section 7)
   - The script should reproduce all artifacts for this phase when run independently
   - If the phase reuses data from a previous phase, the script should load that data
     from the saved artifacts (e.g., `phase1_results.json`) rather than re-querying
4. **Save artifacts** -- all plots, data files go into the experiment directory.
   Name files with phase prefix: `phase1_mass_distribution.png`, `phase2_scaling.png`, etc.
5. **Update phases.json** -- mark the current phase complete, record its artifacts:
   ```json
   {
     "phase": 1,
     "query": "...",
     "status": "completed",
     "summary": "<1-2 sentence summary of findings>",
     "artifacts": ["phase1_code.py", "phase1_plot1.png", "phase1_results.json"]
   }
   ```
6. **Update the report** -- after completing the phase, update `report.tex` and recompile.
   See the "Incremental Report" section below for details.
7. **Present results** -- show key findings concisely
8. **Ask what's next** -- use AskUserQuestion to prompt the user. Suggest 2-3 natural
   follow-up directions based on what was found, plus a "Finish session" option.
   Example:
   ```
   "Based on these results, what would you like to explore next?"
   Options:
   - "Investigate the outliers in the M-T relation"
   - "Compare gas fractions across mass bins"
   - "Look at literature for similar findings"
   - "Finish session"
   ```
9. **Start next phase** -- when the user responds, add a new phase entry to `phases.json`
   and repeat from step 1.

## Incremental Report

The report is written and compiled **after every phase**, not just at the end. This means the
user always has a current PDF reflecting all work done so far.

### After Phase 1 (Initial Creation)

Write `report.tex` and `report.bib` with:
- LaTeX preamble with tcolorbox/xcolor packages and the `querybox` environment (see `/report` skill)
- Title, abstract (preliminary -- can be brief)
- Introduction with the initial research question in a `\begin{querybox}...\end{querybox}`
- Phase 1 section with results, figures, and discussion
- Preliminary conclusions
- Bibliography

Compile: `cd <experiment_dir> && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex`

### After Phase N > 1 (Incremental Update)

Read the existing `report.tex`, then update it:
1. **Add a new section** for the new phase, with its query in a `querybox`, results, and figures
   - Insert the new section before `\section{Summary` or `\section{Conclusions`
2. **Update the abstract** to reflect findings from the new phase
3. **Update the conclusions** to incorporate new findings
4. **Add any new bibliography entries** to `report.bib`
5. Recompile the PDF

This way, `report.pdf` is always up-to-date after every phase.

### LaTeX Query Box Setup (same as /report)

```latex
\usepackage{tcolorbox}
\usepackage{xcolor}

\newtcolorbox{querybox}{
  colback=red!5!white,
  colframe=red!75!black,
  fonttitle=\bfseries,
  title=Research Query,
  sharp corners,
  boxrule=1pt,
  left=6pt, right=6pt, top=4pt, bottom=4pt
}
```

### Section Format Per Phase

```latex
\section{Phase N: <Descriptive Title>}

\begin{querybox}
\textcolor{red!80!black}{<The user's query for this phase>}
\end{querybox}

\noindent <Analysis description, results, discussion...>

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{phaseN_plot.png}
\caption{...}
\label{fig:phaseN_plot}
\end{figure}
```

## When the User Says "Finish"

1. Mark the final phase as completed in `phases.json`
2. Do a final polish of `report.tex`:
   - Refine the abstract to be a proper summary of all phases
   - Ensure conclusions synthesize all findings
   - Check all figures are included with proper captions
   - Verify bibliography is complete
3. Recompile the PDF one final time
4. Report the session summary and PDF path to the user

## Key Principles

- **Short phases are fine** -- a phase can be as simple as "check how many halos have M > 1e15"
  or as complex as "run an MCMC fit to the scaling relation"
- **Always ask before proceeding** -- never start a new analysis direction without user input
- **Track everything** -- every phase's query, code, results, and artifacts are logged
- **Code is mandatory** -- every phase must have a `phase<N>_code.py` that reproduces its results
- **Report is always current** -- after every phase, the PDF reflects all work done so far
- **Build on previous phases** -- reference earlier results, reuse loaded data
- **Suggest intelligently** -- the follow-up suggestions should be informed by the actual results,
  not generic. If you found outliers, suggest investigating them. If a scaling relation looks
  tight, suggest comparing to literature.
