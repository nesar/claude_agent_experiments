---
name: report
description: Generate a LaTeX report from experiment results. Compiles all plots, analysis results, and methodology into a publication-quality PDF. Supports both single-experiment and multi-phase research sessions.
allowed-tools: Read, Write, Glob, Bash(pdflatex *), Bash(bibtex *), Bash(ls *), Bash(cd *)
argument-hint: [experiment directory or description]
---

# Generate LaTeX Report

Context: $ARGUMENTS

## Steps

1. Identify the experiment directory (should be under `runs/experiment_<timestamp>/` or `runs/research_<timestamp>/`)
2. Check if `phases.json` exists -- this determines the report format:
   - **If phases.json exists** (multi-phase research session): use the multi-phase format
   - **If no phases.json** (single experiment): use the standard format
3. Read `prompt.txt` for the initial research question
4. Inventory all generated artifacts:
   - PNG plots and visualizations
   - Results files (CSV, JSON)
   - Methodology notes (methodology.md)
   - Analysis scripts (code.py)

## LaTeX Setup (Both Formats)

The report MUST include these packages and the red query box command:

```latex
\usepackage{tcolorbox}
\usepackage{xcolor}

% Research query box -- used for all user questions/prompts
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

## Standard Format (Single Experiment)

Write `report.tex` with:
- The initial research question from `prompt.txt` displayed in a `\begin{querybox}...\end{querybox}` at the start of the Introduction
- Abstract summarizing the analysis and key findings
- Introduction with scientific context and references
- Methodology section with equations and approach
- Results section with ALL generated plots as figures with detailed captions
- Discussion of findings
- Conclusions
- Complete bibliography with DOI links and arXiv numbers

Example for the query box in Introduction:
```latex
\section{Introduction}

\begin{querybox}
\textcolor{red!80!black}{What is the mass-temperature scaling relation for massive clusters
in the SCIDAC hydrodynamical simulations?}
\end{querybox}

\noindent We investigate the scaling relation between...
```

## Multi-Phase Format (Research Session with phases.json)

Write `report.tex` organized by phases:
- Abstract summarizing the full research session and key findings across all phases
- Introduction with the initial question in a `querybox`
- **One section per phase**, each starting with that phase's query in a `querybox`:
  ```latex
  \section{Phase 2: Gas Fraction Analysis}

  \begin{querybox}
  \textcolor{red!80!black}{How do gas fractions vary across the mass range? Are there outliers?}
  \end{querybox}

  \noindent Following the initial mass-temperature analysis, we examined...
  ```
- Each phase section includes its own results, figures, and discussion
- A final "Summary \& Conclusions" section synthesizing findings across all phases
- Complete bibliography

## Compilation

5. Write bibliography to `report.bib`
6. Compile (must `cd` to the experiment directory first for figure paths):
   ```
   cd /path/to/experiment_dir && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
   ```
7. Verify:
   - PDF compiled successfully
   - ALL PNG files are included as figures
   - No missing references or citations
   - All query boxes render correctly
