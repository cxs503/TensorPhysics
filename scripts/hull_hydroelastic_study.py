"""Run the KCS / KVLCC2 springing-whipping study and write artefacts.

Usage::

    PYTHONPATH=src python3 scripts/hull_hydroelastic_study.py [--out DIR]

Produces ``results.json`` (every acceptance-gate number, programmatic - never
typed in) and the report figures.  Long run (~1-2 min per hull); the JSON is
written before the figures so a figure failure cannot lose the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):  # allow direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from tensorphysics.hull_hydroelastic import KCS, KVLCC2, hull_cases  # noqa: E402
from tensorphysics.hull_hydroelastic.response import (  # noqa: E402
    springing_stress,
    whipping_response,
)
from tensorphysics.hull_hydroelastic.runner import (  # noqa: E402
    added_mass_gate,
    dry_mode_analytic_gate,
    run_case,
)
from tensorphysics.hull_hydroelastic.wet import (  # noqa: E402
    added_mass_distribution,
    ship_beam_properties,
    wet_modes,
)
from tensorphysics.hull_hydroelastic.timoshenko import TimoshenkoBeam  # noqa: E402


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")


def run(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gates": {},
        "cases": {},
    }
    print("[1/4] analytic dry-mode gate", flush=True)
    payload["gates"]["dry_modes"] = dry_mode_analytic_gate()
    print("[2/4] added-mass gate", flush=True)
    payload["gates"]["added_mass"] = added_mass_gate()
    for case in hull_cases():
        print(f"[3/4] case {case.name}", flush=True)
        res = run_case(case)
        # extra diagnostics that need the wet-mode object directly
        ship = case.girder()
        beam = TimoshenkoBeam(ship_beam_properties(ship, n_elements=case.n_elements))
        am = added_mass_distribution(
            ship,
            n_stations=9,
            section_exponent=case.section_exponent,
            n_panels=80,
            x=beam.props.x,
        )
        wet = wet_modes(
            beam,
            am["a33"],
            n_modes=10,
            zeta_structural=0.02,
            b_waterplane=ship.waterplane_width(beam.props.x),
        )
        speed = case.design_speed_kn * 0.514444
        spr = springing_stress(ship, wet, 6.0, 9.0, speed=speed)
        wav = whipping_response(ship, wet, peak_force=5.0e6)
        res["diagnostics"] = {
            "dry_shapes": {
                "x": wet.x.tolist(),
                "modes": wet.dry.shapes[:4].tolist(),
            },
            "wet_shapes": {"modes": wet.wet_shapes[:4].tolist()},
            "a33_profile": {
                "x": wet.x.tolist(),
                "a33": wet.a33.tolist(),
                "mass_per_length": ship.mass_per_length(wet.x).tolist(),
            },
            "modal_stiffness_diagonal": np.diag(wet.modal_stiffness).tolist(),
            "modal_mass_diagonal": np.diag(wet.modal_mass).tolist(),
            "buoyancy_share": wet.buoyancy_share.tolist(),
            "whipping_history": {
                "t": wav["t_s"][::5],
                "stress_MPa": [v / 1.0e6 for v in wav["stress_Pa"][::5]],
            },
            "whipping_modal_peaks_MPa": [v / 1.0e6 for v in wav["modal_peak_stress_Pa"]],
            "speed_ms": speed,
        }
        res["diagnostics"]["springing"] = {
            "omega": spr["omega"],
            "S_zeta": spr["spectrum_m2s"],
            "S_sigma_MPa2s": [v / 1.0e12 for v in spr["stress_spectrum_Pa2s"]],
        }
        payload["cases"][case.name] = res
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(payload, fh, indent=1, default=_json_default)
    payload["elapsed_s"] = time.time() - t0
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=_json_default)
    print("[4/4] done", round(payload["elapsed_s"], 1), "s", flush=True)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("artifacts", "m5549"))
    args = ap.parse_args()
    run(args.out)