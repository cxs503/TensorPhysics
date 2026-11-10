#!/usr/bin/env python
"""Run the full Bayesian MCMC ice–ship friction coefficient calibration pipeline.

Generates:
- artifacts/m9218_calibration.json  (full result)
- artifacts/m9218_chain_samples.npy  (posterior samples)
- artifacts/m9218_posterior_plot.png  (posterior distribution figure)
- artifacts/m9218_trace_plot.png  (MCMC trace plot)
- artifacts/m9218_diagnostics.json  (diagnostics summary)
- runs/m9218/  (run metadata)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensorphysics.ice.mcmc_calibration import (
    MU_HULL_ICE,
    generate_synthetic_data,
    log_likelihood,
    log_prior_normal,
    log_prior_uniform,
    run_full_calibration,
    serialize_result,
    compute_artifact_sha256,
)
from tensorphysics.ice.resistance import ShipGeometry


def main():
    seed = 42
    mu_true = 0.15
    n_steps = 30000
    burn_in = 5000
    thin = 1
    n_chains = 4

    print("=" * 60)
    print("Bayesian MCMC Ice–Ship Friction Coefficient Calibration")
    print("=" * 60)
    print(f"Seed: {seed}")
    print(f"True μ: {mu_true}")
    print(f"Default μ (Lindqvist): {MU_HULL_ICE}")
    print(f"MCMC steps: {n_steps}, burn-in: {burn_in}, thin: {thin}")
    print(f"Chains: {n_chains}")
    print()

    # Setup
    ship = ShipGeometry(
        name="PC6 reference",
        L_pp=150.0,
        B=24.0,
        T=8.5,
        bow_angle_deg=30.0,
        flare_deg=25.0,
    )

    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(
        mu_true=mu_true,
        V_values=np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]),
        h_values=np.array([0.5, 0.8, 1.0, 1.2, 1.5]),
        sigma_noise=5000.0,
        seed=123,
        ship=ship,
    )
    print(f"  Observations: {data['n_observations']}")
    print(f"  Speed range: [{data['V'].min():.1f}, {data['V'].max():.1f}] m/s")
    print(f"  Thickness range: [{data['h'].min():.1f}, {data['h'].max():.1f}] m")
    print()

    # Run calibration
    print("Running MCMC calibration...")
    result = run_full_calibration(
        data["V"],
        data["h"],
        data["R_exp"],
        data["sigma"],
        mu_true=mu_true,
        n_chains=n_chains,
        seed=seed,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        ship=ship,
    )
    print("  Done.")
    print()

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Posterior mean μ:     {result.mu_posterior_mean:.6f}")
    print(f"Posterior std μ:      {result.mu_posterior_std:.6f}")
    print(f"95% CI:               [{result.mu_95_ci_lower:.6f}, {result.mu_95_ci_upper:.6f}]")
    print(f"True μ:               {result.mu_true}")
    print(f"Default μ (Lindqvist): {result.mu_default}")
    print(f"Deviation from default: {result.deviation_from_default:.6f}")
    print(f"Gelman–Rubin R̂:       {result.gelman_rubin_rhat:.6f}")
    print(f"Effective sample size: {result.ess}")
    print(f"Acceptance rate:      {result.acceptance_rate:.4f}")
    print(f"Normalization error:  {result.normalization_error:.10f}")
    print(f"Prior sensitivity:    {'PASS' if result.prior_sensitivity['passed_5pct'] else 'FAIL'}")
    print()

    # Acceptance criteria
    print("=" * 60)
    print("ACCEPTANCE CRITERIA")
    print("=" * 60)
    checks = {
        "μ_true in 95% CI": result.mu_95_ci_lower <= mu_true <= result.mu_95_ci_upper,
        "μ_mean ∈ [0.14, 0.16]": 0.14 <= result.mu_posterior_mean <= 0.16,
        "R̂ < 1.1": result.gelman_rubin_rhat < 1.1,
        "ESS > 1000": result.ess > 1000,
        "Norm error < 0.1%": result.normalization_error < 0.001,
        "Prior sensitivity < 5%": result.prior_sensitivity["passed_5pct"],
    }
    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print(f"\nAll criteria: {'PASS' if all_pass else 'FAIL'}")
    print()

    # Create output directories
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("runs/m9218", exist_ok=True)

    # Save calibration result JSON
    result_json = serialize_result(result)
    result_path = "artifacts/m9218_calibration.json"
    with open(result_path, "w") as f:
        f.write(result_json)
    result_sha = compute_artifact_sha256(result_json.encode("utf-8"))
    print(f"Saved: {result_path} (sha256: {result_sha[:16]}...)")

    # Save chain samples
    samples_path = "artifacts/m9218_chain_samples.npy"
    np.save(samples_path, result.chain_samples)
    samples_sha = compute_artifact_sha256(open(samples_path, "rb").read())
    print(f"Saved: {samples_path} (sha256: {samples_sha[:16]}...)")

    # Save diagnostics
    diagnostics = {
        "mission_id": "m9218",
        "campaign_node": "P519-N05",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "mu_true": mu_true,
        "mu_default": MU_HULL_ICE,
        "n_observations": int(data["n_observations"]),
        "n_chains": n_chains,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "thin": thin,
        "mu_posterior_mean": result.mu_posterior_mean,
        "mu_posterior_std": result.mu_posterior_std,
        "mu_95_ci_lower": result.mu_95_ci_lower,
        "mu_95_ci_upper": result.mu_95_ci_upper,
        "gelman_rubin_rhat": result.gelman_rubin_rhat,
        "ess": result.ess,
        "acceptance_rate": result.acceptance_rate,
        "normalization_error": result.normalization_error,
        "prior_sensitivity_passed": result.prior_sensitivity["passed_5pct"],
        "prior_sensitivity_deviation": result.prior_sensitivity["relative_deviation"],
        "deviation_from_default": result.deviation_from_default,
        "all_criteria_passed": all_pass,
        "result_sha256": result_sha,
        "samples_sha256": samples_sha,
    }
    diag_path = "artifacts/m9218_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved: {diag_path}")

    # Save run metadata
    run_meta = {
        "mission_id": "m9218",
        "branch": "agent/m9218",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_pass else "FAIL",
        "artifacts": [result_path, samples_path, diag_path],
    }
    run_path = "runs/m9218/run_meta.json"
    with open(run_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Saved: {run_path}")

    # Generate figures
    print("\nGenerating figures...")

    # Figure 1: Posterior distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    samples = result.chain_samples
    ax.hist(samples, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white", label="Posterior")
    ax.axvline(mu_true, color="red", linestyle="--", linewidth=2, label=f"True μ = {mu_true}")
    ax.axvline(MU_HULL_ICE, color="green", linestyle="--", linewidth=2, label=f"Default μ = {MU_HULL_ICE}")
    ax.axvline(result.mu_95_ci_lower, color="orange", linestyle=":", linewidth=1.5, label="95% CI bounds")
    ax.axvline(result.mu_95_ci_upper, color="orange", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Ice–Ship Friction Coefficient μ", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Bayesian Posterior Distribution of Ice–Ship Friction Coefficient μ\n"
                 f"μ_post = {result.mu_posterior_mean:.4f} ± {result.mu_posterior_std:.4f}, "
                 f"95% CI = [{result.mu_95_ci_lower:.4f}, {result.mu_95_ci_upper:.4f}]",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    posterior_path = "artifacts/m9218_posterior_plot.png"
    fig.savefig(posterior_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {posterior_path}")

    # Figure 2: MCMC trace plot (4 chains)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        # Re-run individual chains for trace plot
        from tensorphysics.ice.mcmc_calibration import run_metropolis_hastings
        chain = run_metropolis_hastings(
            data["V"], data["h"], data["R_exp"], data["sigma"],
            mu_init=[0.05, 0.10, 0.20, 0.25][i],
            seed=seed + i * 1000,
            n_steps=n_steps,
            burn_in=0,
            thin=1,
            ship=ship,
        )
        ax.plot(chain.chain, alpha=0.7, linewidth=0.5, label=f"Chain {i+1}")
        ax.axhline(mu_true, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("μ", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("MCMC Iteration", fontsize=12)
    fig.suptitle("MCMC Trace Plots (4 Chains)\n"
                 f"R̂ = {result.gelman_rubin_rhat:.4f}, ESS = {result.ess}",
                 fontsize=13)
    fig.tight_layout()
    trace_path = "artifacts/m9218_trace_plot.png"
    fig.savefig(trace_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {trace_path}")

    # Figure 3: Resistance vs Speed comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    V_plot = np.linspace(0.5, 5.5, 100)
    h_plot = 1.0
    R_true = [lindqvist_resistance_wrapper(mu_true, v, h_plot, ship=ship) for v in V_plot]
    R_default = [lindqvist_resistance_wrapper(MU_HULL_ICE, v, h_plot, ship=ship) for v in V_plot]
    R_posterior = [lindqvist_resistance_wrapper(result.mu_posterior_mean, v, h_plot, ship=ship) for v in V_plot]

    ax.plot(V_plot, np.array(R_true)/1e6, "r--", linewidth=2, label=f"True (μ={mu_true})")
    ax.plot(V_plot, np.array(R_default)/1e6, "g--", linewidth=2, label=f"Default (μ={MU_HULL_ICE})")
    ax.plot(V_plot, np.array(R_posterior)/1e6, "b-", linewidth=2, label=f"Posterior mean (μ={result.mu_posterior_mean:.4f})")

    # Plot observations at h=1.0
    mask = np.isclose(data["h"], h_plot)
    ax.scatter(data["V"][mask], data["R_exp"][mask]/1e6, c="black", s=50, zorder=5, label="Synthetic observations")

    ax.set_xlabel("Ship Speed V [m/s]", fontsize=12)
    ax.set_ylabel("Ice Resistance R [MN]", fontsize=12)
    ax.set_title("Ice Resistance vs Speed (h = 1.0 m)\n"
                 "Comparison: True vs Default vs Posterior Mean",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    resistance_path = "artifacts/m9218_resistance_plot.png"
    fig.savefig(resistance_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {resistance_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"All acceptance criteria: {'PASS' if all_pass else 'FAIL'}")
    print(f"Artifacts saved to: artifacts/")
    print(f"Run metadata saved to: runs/m9218/")

    return 0 if all_pass else 1


if __name__ == "__main__":
    # Import lindqvist_resistance_wrapper for the resistance plot
    from tensorphysics.ice.mcmc_calibration import lindqvist_resistance_wrapper
    sys.exit(main())
