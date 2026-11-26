"""Compare both sides of two storing contacts to a continuum harmonic transfer solution."""
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
import sph_layered_storing_contacts as slab


def reference(omega, walls, conductivity, bulk_capacity, resistances, sheet_capacities):
    # Advance (T,q) in +x with q=-k*dT/dx and exp(i*omega*t).
    def advance(left_power):
        temp, power = walls[0], left_power
        sheets = []
        for i, (k, cv) in enumerate(zip(conductivity, bulk_capacity)):
            wave = np.sqrt(1j*omega*cv/k)
            co, si = np.cosh(wave/4), np.sinh(wave/4)
            temp, power = co*temp-si*power/(k*wave), co*power-k*wave*si*temp
            if i < len(resistances):
                r, c = resistances[i], sheet_capacities[i]
                temp -= .5*r*power
                incoming = power
                power -= 1j*omega*c*temp
                if c:
                    sheets.append((temp, incoming, power))
                temp -= .5*r*power
        return temp, np.asarray(sheets)

    offset, _ = advance(0.)
    response, _ = advance(1.)
    west_power = (walls[1]-offset)/(response-offset)
    end, values = advance(west_power)
    if abs(end-walls[1]) > 1e-10:
        raise ArithmeticError("transfer solution missed the prescribed east temperature")
    return values


def measure(n, omega, backend):
    material, positions, _, _, _, _, _, capacities = slab.layered_materials(n)
    walls = (1., .3j)
    # The original contacts (.1, .3, .2) are in series with the two sheets (.4, .7).
    expected = reference(omega, walls, (1., 1., 2.5, 2.5), (1.5, 1.5, .75, .75),
                         (.5, .3, .9), (capacities[0], 0., capacities[1]))
    field, cuts = slab.heat.solve_harmonic_layered_slab(
        torch.zeros(n+4), 1/n, walls, omega, **material, return_x_flux=True, backend=backend)
    measured = np.array([(complex(field[p:p+2].mean()), complex(cuts[p]), complex(cuts[p+2]))
                         for p in positions])
    # Remove sheet particles, retaining both material fluxes at each contact.
    bulk = torch.ones(n+4, dtype=torch.bool)
    for p in positions:
        bulk[p:p+2] = False
    bulk_flux = torch.stack((cuts[:-1], cuts[1:]), dim=-1)[bulk]
    left, right = slab.heat.layered_slab_contact_temperatures(
        field[bulk], bulk_flux, 1/n, material["conductivity"][bulk])
    contacts = [n//4-1, 3*n//4-1]
    recovered = np.column_stack((left[contacts].numpy(), right[contacts].numpy()))
    half_r = np.array([.5, .9])/2
    exact_traces = np.column_stack((expected[:, 0]+half_r*expected[:, 1],
                                    expected[:, 0]-half_r*expected[:, 2]))
    ratio = measured/expected
    balance = measured[:, 1]-measured[:, 2]-1j*omega*np.asarray(capacities)*measured[:, 0]
    return dict(bulk_columns=n, omega=omega, backend=backend,
                max_material_trace_error_K=float(abs(recovered-exact_traces).max()),
                relative_sheet_temperature_error=float(abs(ratio[:, 0]-1).max()),
                relative_face_power_error=float(abs(ratio[:, 1:]-1).max()),
                reference_amplitudes=np.abs(expected).tolist(),
                reference_phases_degrees=np.angle(expected, deg=True).tolist(),
                measured_amplitudes=np.abs(measured).tolist(),
                measured_phases_degrees=np.angle(measured, deg=True).tolist(),
                relative_amplitude_errors=np.abs(np.abs(ratio)-1).tolist(),
                phase_errors_degrees=np.angle(ratio, deg=True).tolist(),
                max_storage_balance_error=float(abs(balance).max()))


if __name__ == "__main__":
    runs = []
    for backend in ("torch", "banded"):
        for omega in (1., 8., 24.):
            previous = None
            for n in (16, 32, 64):
                row = measure(n, omega, backend)
                keys = ("relative_sheet_temperature_error", "relative_face_power_error", "max_material_trace_error_K")
                orders = {} if previous is None else {k: math.log2(previous[k]/row[k]) for k in keys}
                row.update(orders=orders, passed=row["max_storage_balance_error"] < 1e-10 and
                           (n < 64 or (min(orders.values()) > 1.75 and max(row[k] for k in keys) < .02)))
                runs.append(row)
                previous = row
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs,
                          error_columns=["sheet temperature", "left heat flux", "right heat flux"],
                          sources=[provenance(p) for p in (__file__, slab.__file__, slab.heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_harmonic_storing_contacts.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
