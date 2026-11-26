"""Run python examples/resistive_decay_interface_comparison.py with the three solver checkouts installed.

Two half-length rods of unit area have equal diffusivities and a finite contact
resistance. Compare the mean temperatures on either side and the heat leaving
the west rod, over two consecutive time windows.
"""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph


WAVE = 1.5*np.pi
WINDOWS = np.array([0., .025, .05])
RESISTANCE = 4/(3*WAVE)


def cell_data(n):
    x = (np.arange(n)+.5)/n
    k = np.repeat([1., 3.], n//2)
    samples = np.where(x < .5, np.sin(WAVE*x), -np.sin(WAVE*(1-x))/3)
    contacts = np.zeros(n-1)
    contacts[n//2-1] = RESISTANCE
    return k, samples, contacts


def fem(n, times):
    mid = n//2
    x = torch.linspace(0., .5, mid+1, dtype=torch.float64)
    edge = torch.arange(mid)
    conn = torch.stack((edge, edge+1), dim=1)
    pair = torch.tensor([mid, mid+1])
    k = x.new_tensor([1., 3.]).repeat_interleave(mid)
    model = thermal.ThermalModel(
        torch.cat((x, x+.5))[:, None], torch.cat((conn, conn+mid+1)), k, 1., k,
        torch.tensor([0, n+1]), x.new_zeros(2), contact_pairs=pair[None],
        contact_conductance=1/RESISTANCE)
    mode = torch.sin(WAVE*x)
    initial = torch.cat((mode, -mode.flip((0,))/3))
    result = thermal.solve_transient_thermal(model, x.new_tensor(times), initial, method="sdirk2")
    return (result.interval_temperature[:, pair].numpy(),
            thermal.thermal_contact_pair_heat_steps(model, result)[:, 0].numpy())


def fvm(n, times):
    k, samples, contacts = cell_data(n)
    # The FVM initial degrees of freedom represent cell averages.
    initial = samples*np.sinc(WAVE/(2*np.pi*n))
    _, _, _, _, heat, means = rod.solve_transient_stepped_rod(
        np.full(n, 1/n), 1., k, k, [0., 0.], times, initial, contact_resistance=contacts,
        method="sdirk2", flux_order=2, return_step_heat=True, return_interval_temperatures=True)
    cut = n//2-1
    traces = np.column_stack([means[key][:, cut] for key in ("trace_left", "trace_right")])
    return traces, heat["contact"][:, cut]


def particles(n, times):
    k, samples, contacts = cell_data(n)
    _, heat, means = sph.solve_transient_layered_slab(
        samples, times, 1/n, (0., 0.), conductivity=k, volumetric_heat_capacity=k,
        contact_resistance_x=contacts, method="exponential", return_x_heat=True,
        return_interval_temperature=True)
    flux = heat.numpy()/np.diff(times)[:, None]
    traces = sph.layered_slab_contact_temperatures(means, flux, 1/n, k)
    return np.column_stack([side[:, n//2-1] for side in traces]), heat[:, n//2].numpy()


def run():
    # Both sine pieces decay as exp(-w²t). The choice of resistance makes
    # their jump equal to R times their common rightward Fourier flux.
    dt = np.diff(WINDOWS)
    integral = np.exp(-WAVE**2*WINDOWS[:-1])*(-np.expm1(-WAVE**2*dt))/WAVE**2
    reference_means = (integral/dt)[:, None]*np.sin(WAVE/2)*np.array([1., -1/3])
    reference_heat = integral*(-WAVE*np.cos(WAVE/2))
    rows = []
    for name, solve in (("TensorFEM", fem), ("TensorFVM", fvm), ("TensorSPH", particles)):
        previous = None
        for n, steps in ((12, 24), (24, 48), (48, 192)):
            if solve is particles:
                steps = 1
            times = np.linspace(WINDOWS[0], WINDOWS[-1], 2*steps+1)
            traces, transfer = solve(n, times)
            means = traces.reshape(2, steps, 2).mean(axis=1)
            heat = transfer.reshape(2, steps).sum(axis=1)
            error = np.r_[abs(means-reference_means).max(axis=0), max(abs(heat-reference_heat))]
            passed = ((previous is None or np.all(previous > 3*error))
                      and (n < 48 or np.all(error < [1e-3, 1e-3, 1e-4])))
            rows.append(dict(solver=name, resolution=n, steps_per_window=steps,
                             mean_west_east_K=means.tolist(), rightward_heat_J_m2=heat.tolist(),
                             max_trace_errors_K=error[:2].tolist(), max_heat_error_J_m2=error[2],
                             passed=bool(passed)))
            previous = error
    return dict(windows_s=WINDOWS.tolist(), contact_resistance_m2K_W=RESISTANCE,
                reference_mean_west_east_K=reference_means.tolist(),
                reference_rightward_heat_J_m2=reference_heat.tolist(), runs=rows,
                sources=[provenance(p) for p in (__file__, thermal.__file__, rod.__file__, sph.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
