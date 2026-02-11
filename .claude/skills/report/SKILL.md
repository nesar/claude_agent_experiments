---
name: report
description: Generate a LaTeX report from experiment results. Compiles all plots, analysis results, and methodology into a publication-quality PDF. Use after completing an analysis in an experiment directory.
allowed-tools: Read, Write, Glob, Bash(pdflatex *), Bash(bibtex *), Bash(ls *)
argument-hint: [experiment directory or description]
---

# Generate LaTeX Report

Context: $ARGUMENTS

## Steps

1. Identify the experiment directory (should be under `runs/experiment_<timestamp>/`)
2. Inventory all generated artifacts:
   - PNG plots and visualizations
   - Results files (CSV, JSON)
   - Methodology notes (methodology.md)
   - Analysis scripts (code.py)
3. Write a LaTeX report (`report.tex`) in physics journal style:
   - Abstract summarizing the analysis and key findings
   - Introduction with scientific context and references
   - Methodology section with equations and approach
   - Results section with ALL generated plots as figures with detailed captions
   - Discussion of findings
   - Conclusions
   - Complete bibliography with DOI links and arXiv numbers
4. Write bibliography to `report.bib`
5. Compile: `pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex`
6. Verify:
   - PDF compiled successfully
   - ALL PNG files are included as figures
   - No missing references or citations
