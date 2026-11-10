#!/usr/bin/env python3
"""Benchmark: pressure melting–refreezing bond cycle.

Runs a representative ice–structure contact scenario and outputs
machine-checkable JSON with all acceptance criteria results.
"""

import json
import math
import os
import sys

# Ensure src is on the path
# The benchmark is at benchmarks/pending/pressure_melting_contact/run_benchmark.py
# The src directory is at the repo root: ../../.. from this file
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_src_path = os.path.join(_repo_root, "src")
sys.path.insert(0, _src_path)

from tensorphysics.ice_contact.pressure_melting import (
    pressure_melting_threshold as pm_exact,
)
from tensorphysics.ice_contact.pressure_melting_cycle import (
    ContactStateEnum,
    PressureMeltingCycle,
    PressureMeltingCycleParams,
    bond_strength,
    contact_force,
    pressure_melting_threshold_linear,
    refreeze_time_constant,
)
from tensorphysics.ice_contact.properties import IceWaterProperties

PROPS = IceWaterProperties()


def run_benchmark():
    """Run the full benchmark and return results dict."""
    results = {}

    # --- C1: P_melt vs Clapeyron ---
    ice_temp = 263.15  # -10°C
    p_linear = pressure_melting_threshold_linear(ice_temp, PROPS)
    p_exact_val = pm_exact(ice_temp, PROPS)
    c1_error = abs(p_linear - p_exact_val) / p_exact_val
    results["C1"] = {
        "description": "P_melt vs Clapeyron exact",
        "p_linear_Pa": p_linear,
        "p_exact_Pa": p_exact_val,
        "relative_error": c1_error,
        "threshold": 0.01,
        "pass": c1_error < 0.01,
    }

    # --- C2: No-melt degradation ---
    params = PressureMeltingCycleParams()
    cycle = PressureMeltingCycle(params)
    p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)
    f_dry = params.k_n * 1e-5
    for _ in range(100):
        r = cycle.step(1e-5, 0.5 * p_melt, 0.1, ice_temp, 1e-5)
    f_actual = r["f_contact"]
    c2_dev = abs(f_actual - f_dry) / f_dry
    results["C2"] = {
        "description": "No-melt degradation: P < P_melt → h=0, force deviation",
        "h_water": cycle.h_water,
        "f_dry_Pa": f_dry,
        "f_actual_Pa": f_actual,
        "relative_deviation": c2_dev,
        "threshold": 0.001,
        "pass": c2_dev < 0.001 and cycle.h_water == 0.0,
    }

    # --- C3: Water film positivity ---
    cycle2 = PressureMeltingCycle(params)
    h_min = float("inf")
    for _ in range(200):
        cycle2.step(1e-5, 2.0 * p_melt, 1.0, ice_temp, 1e-5)
        h_min = min(h_min, cycle2.h_water)
    results["C3"] = {
        "description": "Water film positivity: h_water >= 0",
        "h_water_min": h_min,
        "pass": h_min >= 0.0,
    }

    # --- C4: Contact force monotonicity ---
    h_vals = [i * 1e-6 / 50 for i in range(51)]
    f_vals = [contact_force(1e9, 1e-5, h, 1e-6) for h in h_vals]
    monotone = all(f_vals[i] <= f_vals[i - 1] + 1e-12 for i in range(1, len(f_vals)))
    results["C4"] = {
        "description": "Contact force monotonicity: h_water ↑ → F ↓",
        "monotone": monotone,
        "pass": monotone,
    }

    # --- C5: Refreezing rate monotonicity ---
    temps = [270.15 - i * 0.5 for i in range(40)]
    taus = [refreeze_time_constant(t, PROPS) for t in temps]
    tau_monotone = all(taus[i] < taus[i - 1] for i in range(1, len(taus)))
    results["C5"] = {
        "description": "Refreezing rate monotonicity: T_env ↓ → τ_refreeze ↓",
        "tau_monotone": tau_monotone,
        "pass": tau_monotone,
    }

    # --- C6: Bond strength limits ---
    sigma_0 = 1e6
    tau_ref = 0.1
    sigma_t0 = bond_strength(0.0, tau_ref, sigma_0)
    sigma_t10 = bond_strength(10 * tau_ref, tau_ref, sigma_0)
    c6_pass = sigma_t0 == 0.0 and abs(sigma_t10 - sigma_0) / sigma_0 < 1e-4
    results["C6"] = {
        "description": "Bond strength limits: t=0→σ=0, t→∞→σ→σ_0",
        "sigma_t0": sigma_t0,
        "sigma_t10tau": sigma_t10,
        "sigma_0": sigma_0,
        "pass": c6_pass,
    }

    # --- C7: Peak load reduction ---
    cycle3 = PressureMeltingCycle(params)
    n_pts = 500
    times = [i * 1e-4 for i in range(n_pts)]
    pressures = [2e8 * (0.5 + 0.5 * math.sin(2 * math.pi * t * 5)) for t in times]
    f_pm = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        r = cycle3.step(dt, pressures[i], 0.5, 268.15, 1e-5)
        f_pm.append(r["f_contact"])
    f_peak_pm = max(f_pm)
    f_peak_dry = params.k_n * 1e-5
    c7_reduction = (f_peak_dry - f_peak_pm) / f_peak_dry
    results["C7"] = {
        "description": "Peak load reduction: with PM model F_peak reduced >= 5%",
        "f_peak_dry_Pa": f_peak_dry,
        "f_peak_pm_Pa": f_peak_pm,
        "reduction": c7_reduction,
        "threshold": 0.05,
        "pass": c7_reduction >= 0.05,
    }

    # --- C8: Frequency modulation ---
    cycle4 = PressureMeltingCycle(PressureMeltingCycleParams(tau_drain=5e-4))
    n_pts2 = 1000
    times2 = [i * 5e-5 for i in range(n_pts2)]
    pressures2 = [1.5e8 * (0.5 + 0.5 * math.sin(2 * math.pi * t * 10)) for t in times2]
    f_pm2 = [params.k_n * 1e-5]
    for i in range(1, len(times2)):
        dt = times2[i] - times2[i - 1]
        r = cycle4.step(dt, pressures2[i], 0.5, 268.15, 1e-5)
        f_pm2.append(r["f_contact"])

    def count_crossings(signal):
        threshold = sum(signal) / len(signal)
        return sum(1 for i in range(1, len(signal))
                   if (signal[i - 1] < threshold) != (signal[i] < threshold))

    f_dry2 = [params.k_n * 1e-5 * (0.5 + 0.5 * math.sin(2 * math.pi * t * 10))
              for t in times2]
    n_cross_dry = count_crossings(f_dry2[1:])
    n_cross_pm = count_crossings(f_pm2)
    freq_ratio = n_cross_pm / n_cross_dry if n_cross_dry > 0 else 0.0
    c8_reduction = 1.0 - freq_ratio
    results["C8"] = {
        "description": "Frequency modulation: with PM model frequency reduced >= 10%",
        "crossings_dry": n_cross_dry,
        "crossings_pm": n_cross_pm,
        "frequency_ratio": freq_ratio,
        "reduction": c8_reduction,
        "threshold": 0.10,
        "pass": c8_reduction >= 0.10,
    }

    # --- Summary ---
    all_pass = all(r["pass"] for r in results.values())
    results["summary"] = {
        "total_criteria": 8,
        "passed": sum(1 for r in results.values() if r.get("pass")),
        "failed": sum(1 for r in results.values() if not r.get("pass")),
        "all_pass": all_pass,
    }

    return results


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))

    # Save to artifacts
    out_path = os.path.join(_repo_root, "artifacts", "pressure_melting_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
