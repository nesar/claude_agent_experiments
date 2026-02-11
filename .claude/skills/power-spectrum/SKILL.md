---
name: power-spectrum
description: Run matter power spectrum analysis using mcp-ke. Compute P(k) for cosmological models, compare with observations, run MCMC parameter estimation, and create publication-quality plots.
argument-hint: [analysis description]
---

# Power Spectrum Analysis

Analysis goal: $ARGUMENTS

## Available Tools (via mcp-ke)

### Data & Grid
- `load_observational_data(filepath)` -- load eBOSS DR14 Lyman-alpha data
- `create_theory_k_grid()` -- generate logarithmic k-grid (0.0001-10 h/Mpc)

### Cosmological Models
- `get_lcdm_params()` -- Planck 2018 baseline LCDM
- `get_nu_mass_params(sum_mnu_eV, N_species)` -- LCDM + massive neutrinos
- `get_wcdm_params(w0)` -- dark energy with constant equation of state

### Computation
- `compute_power_spectrum(params, k_values)` -- single model P(k) via CLASS
- `compute_all_models(k_values, models)` -- batch compute standard models
- `compute_suppression_ratios(model_results, k_values, reference_model)` -- P(k)/P_ref(k)

### Visualization
- `plot_power_spectra(k_theory, model_results, k_obs, Pk_obs, Pk_obs_err, save_path)` -- comparison plot
- `plot_suppression_ratios(k_values, suppression_ratios, reference_model, save_path)` -- ratio plot

### MCMC Parameter Estimation
- `run_mcmc_cosmology(param_bounds, k_obs, Pk_obs, Pk_obs_err, ...)` -- affine-invariant MCMC
- `create_mcmc_corner_plot(samples_csv, ...)` -- GetDist corner plot
- `create_mcmc_trace_plot(samples_csv, ...)` -- chain convergence diagnostics
- `analyze_mcmc_samples(samples_csv)` -- extract statistics
- `compute_best_fit_power_spectrum(best_fit_params, k_values)` -- best-fit prediction

## Typical Workflow

1. Load observational data and create theory k-grid
2. Compute power spectra for relevant models
3. Plot comparison with observations
4. If fitting: run MCMC, analyze chains, plot corner/trace
5. Save all outputs to the experiment directory
