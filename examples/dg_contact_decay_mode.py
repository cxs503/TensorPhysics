"""SIPG cell means and numerical face heats for the two-material contact decay mode."""
import json

import numpy as np
import torch
from tensordg import poisson_2d_bc as poisson
from tensordg import transient_diffusion_2d as diffusion

import fem_contact_decay_mode as case
from conduction_provenance import provenance


def measure(n, method):
    model, _, decay, initial_flux = case.build_problem(n)
    cut = int(model.contact_pairs[0, 0])
    interface = cut/n
    k, cv = model.conductivity.numpy(), model.specific_heat.numpy()
    wave = np.sqrt(decay*cv/k)
    resistance = np.zeros((n-1, 2))
    resistance[cut-1] = 1/float(model.contact_conductance)
    options = dict(bc=dict(left="dirichlet", right="dirichlet", bottom="neumann", top="neumann"),
                   conductivity=np.tile(k[:, None], (1, 2)), contact_resistance={"x": resistance})
    def initial(x, y):
        return torch.where(x < interface, torch.sin(wave[0]*x), 2*torch.sin(wave[-1]*(1-x)))
    times = np.linspace(0, 2/decay, n+1)
    _, _, states, means = diffusion.solve_transient_diffusion_2d(
        initial, times, ncells=(n, 2), degree=1, capacity=np.tile(cv[:, None], (1, 2)),
        method=method, theta=.5, return_interval_means=True, **options)
    centers = (np.arange(n)+.5)/n
    amplitude = np.where(centers < interface, np.sin(wave*centers), 2*np.sin(wave*(1-centers)))
    exact = np.exp(-decay*times[:, None])*((amplitude*np.sinc(wave/(2*np.pi*n)))[None, :])
    cells = states[:, :, :, 0, 0].numpy()
    temperature_error = np.max(abs(cells-exact[:, :, None]))
    # Integrate the affine face-power operator over four equal output windows.
    # Zero boundary loads and uniform timesteps permit averaging the stage means.
    report = times[::n//4]
    heats = []
    for dt, field in zip(np.diff(report), means.reshape(4, n//4, *means.shape[1:]).mean(1)):
        powers, _ = poisson.poisson_face_powers_2d(field, **options)
        heats.append(dt*powers[[0, cut, n]].sum(1).numpy())
    heats = np.array(heats)
    change = np.diff(cells[::n//4].mean(axis=2), axis=0)*cv/n
    layer_change = np.stack((change[:, :cut].sum(1), change[:, cut:].sum(1)), axis=1)
    heat_error = max(abs(np.cumsum(heats[:, 1])+initial_flux*np.expm1(-decay*report[1:])/decay))
    return dict(columns=n, time_steps=n, method=method, temperature_error_K=float(temperature_error),
                contact_heat_error_J_m2=float(heat_error),
                layer_balance_J_m2=float(np.max(abs(layer_change+np.diff(heats, axis=1)))))


if __name__ == "__main__":
    rows = []
    for method in ("theta", "sdirk2"):
        previous = None
        for n in (20, 40, 80, 160):
            row = measure(n, method)
            errors = np.array([row["temperature_error_K"], row["contact_heat_error_J_m2"]])
            orders = None if previous is None else np.log2(previous/errors).tolist()
            passed = row["layer_balance_J_m2"] < 1e-9
            if n == 160:
                passed &= min(orders) > 1.8 and max(errors) < 2e-4
            row.update(orders=orders, passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in
        (__file__, case.__file__, poisson.__file__, diffusion.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG:../TensorFEM/src python examples/dg_contact_decay_mode.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
