"""Check interval heat and mean temperatures at two storing contacts during a cycle."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
import sph_harmonic_storing_contacts as harmonic
import sph_layered_storing_contacts as slab


def measure(n, omega, method):
    material, sheets, _, _, _, _, _, capacity = slab.layered_materials(n)
    walls = (1., .3j)
    resistance = np.array([.5, .9])
    reference = harmonic.reference(omega, walls, (1., 1., 2.5, 2.5), (1.5, 1.5, .75, .75),
                                   (.5, .3, .9), (capacity[0], 0., capacity[1]))
    # Start on the discrete periodic orbit, with no separate start-up transient.
    initial = slab.heat.solve_harmonic_layered_slab(
        torch.zeros(n+4), 1/n, walls, omega, **material, backend="banded").real
    drive = tuple(lambda t, a=a: float((a*np.exp(1j*omega*t)).real) for a in walls)
    times = np.linspace(0., 2*np.pi/omega, 4*n+1)
    state, heat, mean = slab.heat.solve_transient_layered_slab(
        initial, times, 1/n, drive, **material, method=method,
        return_x_heat=True, return_interval_temperature=True)
    # Check independently integrated windows of unequal duration on both sides of a reversal.
    limits = [0, n//2, 2*n, 3*n, 4*n]
    bulk = torch.ones(n+4, dtype=torch.bool)
    for p in sheets:
        bulk[p:p+2] = False
    traces = np.column_stack((reference[:, 0]+resistance/2*reference[:, 1],
                              reference[:, 0]-resistance/2*reference[:, 2]))
    rows = []
    for start, stop in zip(limits, limits[1:]):
        duration = times[stop]-times[start]
        factor = np.exp(1j*omega*times[start])*np.expm1(1j*omega*duration)/(1j*omega)
        total = heat[start:stop].sum(dim=0)
        widths = torch.from_numpy(np.diff(times[start:stop+1]))
        average = (mean[start:stop]*widths[:, None]).sum(dim=0)/duration
        energies = np.array([[total[p].item(), total[p+2].item()] for p in sheets])
        stored = np.array([c*(state[stop, p:p+2]-state[start, p:p+2]).mean().item()
                           for p, c in zip(sheets, capacity)])
        flux = torch.stack((total[:-1], total[1:]), dim=-1)[bulk]/duration
        left, right = slab.heat.layered_slab_contact_temperatures(
            average[bulk], flux, 1/n, material["conductivity"][bulk])
        cuts = [n//4-1, 3*n//4-1]
        observed = np.column_stack((left[cuts], right[cuts]))
        rows.append(dict(phase_interval_cycles=[start/(4*n), stop/(4*n)],
                         material_trace_error_K=float(abs(observed-(traces*factor/duration).real).max()),
                         section_heat_error_J_m2=float(abs(energies-(reference[:, 1:]*factor).real).max()),
                         storage_change_error_J_m2=float(abs(stored-
                             (1j*omega*np.asarray(capacity)*reference[:, 0]*factor).real).max()),
                         storage_balance_J_m2=float(abs(energies[:, 0]-energies[:, 1]-stored).max())))
    return rows


if __name__ == "__main__":
    runs = []
    for method in ("theta", "sdirk2"):
        for omega in (1., 8., 24.):
            previous = None
            for n in (16, 32, 64):
                windows = measure(n, omega, method)
                errors = np.array([max(row[key] for row in windows) for key in
                                   ("material_trace_error_K", "section_heat_error_J_m2", "storage_change_error_J_m2")])
                assert max(row["storage_balance_J_m2"] for row in windows) < 1e-9
                orders = np.log2(previous/errors) if previous is not None else None
                if n == 64:
                    np.testing.assert_array_less(1.7, orders)
                runs.append(dict(method=method, omega_rad_s=omega, bulk_columns=n,
                                 trace_heat_storage_orders=orders.tolist() if orders is not None else None, windows=windows))
                previous = errors
    print(json.dumps(dict(runs=runs, sources=[provenance(p) for p in
                        (__file__, harmonic.__file__, slab.__file__, slab.heat.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_storing_contact_intervals.py"),
        indent=2, allow_nan=False))
