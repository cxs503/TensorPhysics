"""Example: maturity API of the underwater attached-ice module (p302w2-n09).

Demonstrates the three capabilities added in the maturity pass:

1. closed-form steady state (``equilibrium_thickness``) and why it is monotone,
2. time-resolved P50/P90 envelope over the uncertain forcing,
3. the thin JSON hand-off consumed by the resistance / propulsion chain,
   including the honest detachment-regime diagnostic.

Run:  python examples/attached_ice/example_handoff_and_envelope.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

# this worktree's src/ (an installed build of another checkout may shadow it)
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorphysics.ice.attached_ice import (
    T_FREEZE_FW, build_handoff, detachment_regime, equilibrium_thickness,
    monte_carlo_envelope, to_json,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def steady_state_table():
    print("u* [m/s]   h_eq [mm]   limited by    tau_wall [Pa]")
    for u_star in (0.01, 0.02, 0.05, 0.10, 0.15):
        eq = equilibrium_thickness(undercooling_k=2.0, u_star=u_star, c_f=1e-4,
                                   t_water_k=T_FREEZE_FW + 0.2)
        print(f"{u_star:8.3f}   {1e3 * eq['h_eq_m']:9.3f}   {eq['limited_by']:12s} "
              f"{eq['tau_wall_Pa']:11.2f}")


def monotonicity_check():
    us = np.linspace(0.005, 0.15, 200)
    h = np.array([equilibrium_thickness(2.0, float(u), c_f=1e-4,
                                        t_water_k=T_FREEZE_FW + 0.2)["h_eq_m"]
                  for u in us])
    print(f"monotonicity over {us.size} points: violations = {int(np.sum(np.diff(h) > 0))}")


def envelope():
    env = monte_carlo_envelope(duration_s=6 * 3600.0, n_steps=150, n_samples=100,
                              seed=20261110, t_water_k=T_FREEZE_FW + 0.2)
    print(f"envelope ({env['n_samples']} samples): h_ice end P50 = "
          f"{1e3 * env['h_ice_final_p50']:.2f} mm, P90 = "
          f"{1e3 * env['h_ice_final_p90']:.2f} mm")
    return env


def handoff():
    payload = build_handoff(
        undercooling_k=2.0,
        duration_s=7200.0,
        speed_m_s=3.0,
        c_f=1e-4,
        water_temperature_k=T_FREEZE_FW + 0.2,
        ice_temperature_c=-5.0,
        cf_total=0.004,
        n_mc=50,
        n_steps=200,
    )
    sc = payload["scalars"]
    print("handoff scalar summary:")
    print(json.dumps({k: sc[k] for k in (
        "h_ice_end_m", "k_s_end_m", "delta_cf_end", "delta_t_over_t_end",
        "delta_t_over_t_linearization_ok")}, indent=2))
    path = os.path.join(OUT, "attached_ice_handoff.json")
    with open(path, "w") as f:
        f.write(to_json(payload))
    print("written:", path)
    return payload


def regime():
    reg = detachment_regime(u_star=0.1, tau_bond=0.5e6)
    print(f"shear detachment active at 3.6 m/s? {reg['shear_detachment_active']} "
          f"(needs {reg['speed_crit_m_s']:.0f} m/s of mean flow)")
    print("=> the active removal mechanism at ship scale is melting, not stripping")


if __name__ == "__main__":
    steady_state_table()
    monotonicity_check()
    envelope()
    handoff()
    regime()