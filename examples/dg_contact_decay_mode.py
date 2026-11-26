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
    # At the contact the degree-one x basis has values (1,+1) on the
    # left and (1,-1) on the right. The constant y coefficient gives
    # the face average for each transverse cell separately.
    left = states[:, cut-1, :, :, 0].sum(dim=-1).numpy()
    right = (states[:, cut, :, 0, 0]-states[:, cut, :, 1, 0]).numpy()
    trace_amplitude = np.array([np.sin(wave[0]*interface), 2*np.sin(wave[-1]*(1-interface))])
    trace_exact = np.exp(-decay*times[:, None, None])*trace_amplitude
    trace_error_history = abs(np.stack((left, right), axis=-1)-trace_exact)
    trace_errors = np.max(trace_error_history, axis=(0, 1))
    jump_error = np.max(abs(left-right-(trace_exact[..., 0]-trace_exact[..., 1])))
    # Integrate the affine face-power operator over four equal output windows.
    # Zero boundary loads and uniform timesteps permit averaging the stage means.
    report = times[::n//4]
    heats = []
    law_residuals = []
    for dt, field in zip(np.diff(report), means.reshape(4, n//4, *means.shape[1:]).mean(1)):
        powers, _ = poisson.poisson_face_powers_2d(field, **options)
        left_mean = field[cut-1, :, :, 0].sum(dim=-1).numpy()
        right_mean = (field[cut, :, 0, 0]-field[cut, :, 1, 0]).numpy()
        # Each face spans half the unit strip. Convert its integrated power
        # to flux density before applying the areal contact resistance.
        law_residuals.append(abs(left_mean-right_mean-resistance[cut-1]*2*powers[cut].numpy()))
        heats.append(dt*powers[[0, cut, n]].sum(1).numpy())
    heats = np.array(heats)
    change = np.diff(cells[::n//4].mean(axis=2), axis=0)*cv/n
    layer_change = np.stack((change[:, :cut].sum(1), change[:, cut:].sum(1)), axis=1)
    heat_error = max(abs(np.cumsum(heats[:, 1])+initial_flux*np.expm1(-decay*report[1:])/decay))
    return dict(columns=n, time_steps=n, method=method, temperature_error_K=float(temperature_error),
                trace_errors_left_right_K=trace_errors.tolist(), jump_error_K=float(jump_error),
                final_trace_errors_left_right_K=trace_error_history[-1].max(axis=0).tolist(),
                max_interval_contact_law_residual_K=float(np.max(law_residuals)),
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
            trace_errors = np.array([*row["trace_errors_left_right_K"], row["jump_error_K"]])
            trace_orders = None if previous is None else np.log2(previous_trace_errors/trace_errors).tolist()
            previous_trace_errors = trace_errors
            passed = row["layer_balance_J_m2"] < 1e-9
            passed &= row["max_interval_contact_law_residual_K"] < 1e-10
            if n == 160:
                passed &= min(orders) > 1.8 and max(errors) < 2e-4
                passed &= min(trace_orders) > 1.8 and max(trace_errors) < 2e-4
            row.update(trace_orders_left_right_jump=trace_orders, orders=orders, passed=bool(passed))
            rows.append(row)
            previous = errors
    passed = all(r["passed"] for r in rows)
    print(json.dumps(dict(passed=passed, runs=rows, sources=[provenance(p) for p in
        (__file__, case.__file__, poisson.__file__, diffusion.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG:../TensorFEM/src python examples/dg_contact_decay_mode.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
