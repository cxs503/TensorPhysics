"""Measure the signed heat impulse from a first-step Dirichlet correction and its feedback."""
import json

import numpy as np
import torch

import lbm_cube_film_corners as cube
from conduction_provenance import provenance


def measure(n, tau, kinds):
    dx, capacity = 1/n, 1.7
    dt = (tau-.5)*dx**2/(4*.08)
    axis = kinds.index("temperature")

    def run(corrected):
        heat, energy, early = [], [], []
        def record(step, fluxes, field):
            heat.append([[float(q.sum())*dx**2*dt for q in pair] for pair in fluxes])
            energy.append(float(field.sum())*dx**3*capacity)
            if step < 2:
                early.append(torch.stack(fluxes[axis]))
        result = cube.measure(n,tau,kinds,correct_first=corrected,observe=record)
        return np.asarray(heat), np.asarray(energy), torch.stack(early), result

    base_heat, base_energy, base_flux, base = run(False)
    new_heat, new_energy, new_flux, corrected = run(True)
    delta = new_heat-base_heat
    early = new_flux-base_flux
    # A temperature shift injects only the incoming population d at that wall node.
    # On the next collision the opposite population acquires w*d/tau, w=1/8.
    # Restoring the original Dirichlet input then gives a flux change -2*w*d/tau
    # in the same units as the first-step flux impulse.
    prediction_error = float((early[1]+early[0]/(4*tau)).abs().max())
    cumulative = np.cumsum(delta.sum(axis=(1,2)))
    energy_error = float(abs(cumulative-(new_energy-base_energy)).max())
    return dict(n=n,tau=tau,wall_types_xyz=kinds,initial_impulse_J=float(delta[0,axis].sum()),
        second_step_dirichlet_heat_change_J=float(delta[1,axis].sum()),
        predicted_second_to_first_ratio=-1/(4*tau),max_second_step_prediction_error_W_m2=prediction_error,
        final_extra_stored_heat_J=float(new_energy[-1]-base_energy[-1]),
        max_cumulative_balance_error_J=energy_error,
        baseline_peak_flux_error_W_m2=base["wall_flux_error_W_m2"],
        corrected_peak_flux_error_W_m2=corrected["wall_flux_error_W_m2"],
        interval_heat_changes_xyz_lower_upper_J=delta.tolist(),
        passed=bool(prediction_error < 1e-12 and energy_error < 1e-13 and early[0].abs().max()>1e-6))


if __name__ == "__main__":
    runs = [measure(n,tau,kinds) for n in (8,16,32) for tau in (.75,1.5) for kinds in
            (("temperature","flux","film"),("film","temperature","flux"),("flux","film","temperature"))]
    print(json.dumps(dict(reference="Second-step inward Dirichlet flux change equals -1/(4*tau) times the local first-step impulse, both signs inward.",
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorLBM/src python examples/lbm_startup_heat_response.py",
        runs=runs,sources=[provenance(p) for p in (__file__,cube.__file__,cube.lbm.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
