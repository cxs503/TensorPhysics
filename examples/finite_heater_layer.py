"""Finite-thickness harmonic heater between dissimilar slabs, compared with its lumped limit."""
import json
from inspect import getfile

import numpy as np
import torch

from conduction_provenance import provenance
from interface_heat_generation import sheet_reference
from tensorfvm.cht import energy
from tensorfem import thermal


def reference(thickness, fraction, heater_k):
    omega, capacity, resistance, drive, wave, sheet, lumped_flux, _ = sheet_reference(fraction)
    # Each outer slab and its contact present a thermal admittance to the heater.
    admittance = 1/(np.tanh(.5*wave)/(wave*[1.,4.])+resistance*np.array([fraction, 1-fraction]))
    left, right = admittance
    lam = np.sqrt(1j*omega*capacity/(thickness*heater_k))
    z = lam*thickness/2
    c, s, g = np.cosh(z), np.sinh(z), heater_k*lam
    particular = drive/(1j*omega*capacity)
    # T = particular + a*cosh(lambda*x) + b*sinh(lambda*x), centered at x=0.
    # Both outward fluxes equal their own admittance times the heater-side trace.
    matrix = [[-g*s-left*c, g*c+left*s], [-g*s-right*c, -g*c-right*s]]
    a, b = np.linalg.solve(matrix, particular*admittance)
    power = admittance*(particular+a*c+np.array([-1, 1])*b*s)
    return np.r_[particular+a*s/z, power], np.r_[sheet, lumped_flux]


def numerical(n, thickness, fraction, heater_k):
    omega, capacity, resistance, drive, *_ = sheet_reference(fraction)
    counts = [n, n//4, n]
    dx = np.repeat([.5/n, thickness/counts[1], .5/n], counts)
    k = np.repeat([1., heater_k, 4.], counts)
    cv = np.repeat([1., capacity/thickness, 2.], counts)
    source = np.repeat([0., drive/thickness, 0.], counts)
    faces = np.array([n-1, n+counts[1]-1])
    contact = np.zeros(len(dx)-1)
    contact[faces] = resistance*np.array([fraction, 1-fraction])
    walls = {side: {"type": "dirichlet", "values": 0.} for side in ("west","east")}
    t, _, power = energy.solve_harmonic_conduction(dx, [1.], k, cv, walls, omega,
                source=source, contact_resistance_x=contact[None, :], return_interior_power=True)
    mean = t[0,n:n+counts[1]].mean()
    outward = power["x"][0,faces]*[-1,1]
    return np.r_[mean, outward], abs(1j*omega*capacity*mean+outward.sum()-drive)


def fem_model(n, thickness, fraction, heater_k):
    omega, capacity, resistance, drive, *_ = sheet_reference(fraction)
    counts = np.array([n, n//4, n])
    bounds = [0., .5, .5+thickness, 1.+thickness]
    # Distinct coincident nodes retain both temperature traces at each contact.
    grids = [np.linspace(a, b, int(m)+1) for a, b, m in zip(bounds[:-1], bounds[1:], counts)]
    nodes = torch.from_numpy(np.concatenate(grids)[:, None])
    starts = np.r_[0, np.cumsum(counts[:-1]+1)]
    left = torch.from_numpy(np.concatenate([start+np.arange(m) for start, m in zip(starts, counts)]))
    conn = torch.stack((left, left+1), dim=1)
    pairs = torch.from_numpy(np.column_stack((starts[1:]-1, starts[1:])))
    model = thermal.ThermalModel(nodes, conn, torch.from_numpy(np.repeat([1., heater_k, 4.], counts)),
                1., torch.from_numpy(np.repeat([1., capacity/thickness, 2.], counts)),
                torch.tensor([0, len(nodes)-1]), nodes.new_zeros(2),
                heat_source=torch.from_numpy(np.repeat([0., drive/thickness, 0.], counts)),
                contact_pairs=pairs, contact_conductance=1/(resistance*nodes.new_tensor([fraction,1-fraction])))
    return model, slice(n, n+counts[1])


def fem_response(n, thickness, fraction, heater_k):
    omega, capacity, _, drive, *_ = sheet_reference(fraction)
    model, heater = fem_model(n, thickness, fraction, heater_k)
    solution = thermal.solve_harmonic_thermal(model, omega, sparse=True)
    t = solution.temperature
    mean = t[model.elements[heater]].mean().item()
    nodal = thermal.thermal_element_nodal_power(model, t, omega).numpy()
    outward = -np.array([nodal[heater.start, 0], nodal[heater.stop-1, 1]])
    # Each outward power enters the adjacent outer slab across a contact link.
    pairs = model.contact_pairs
    link = ((t[pairs[:,0]]-t[pairs[:,1]])*model.contact_conductance).numpy()*[-1,1]
    np.testing.assert_allclose(outward, link, atol=1e-12, rtol=0)
    return np.r_[mean, outward], abs(1j*omega*capacity*mean+outward.sum()-drive)


def main():
    runs, limits = [], []
    heater_k = 2.
    for fraction in (.15, .85):
        thickness = .04
        exact, _ = reference(thickness, fraction, heater_k)
        previous = {}
        for n in (8, 16, 32, 64, 128):
            for name, solve in (("FVM", numerical), ("FEM", fem_response)):
                values, balance = solve(n, thickness, fraction, heater_k)
                errors = abs(values-exact)
                orders = np.log2(previous[name]/errors) if name in previous else None
                passed = np.isfinite(errors).all() and balance < 1e-10
                if orders is not None:
                    passed &= min(orders) > 1.8
                runs.append(dict(solver=name, left_resistance_fraction=fraction, thickness_m=thickness,
                              outer_cells_per_side=n, heater_cells=n//4, mean_temperature_error_K=errors[0],
                              outward_flux_errors_W_m2=errors[1:].tolist(), balance_W_m2=balance,
                              orders=None if orders is None else orders.tolist(), passed=bool(passed)))
                previous[name] = errors
        previous = None
        for thickness in (.08, .04, .02, .01, .005):
            exact, lumped = reference(thickness, fraction, heater_k)
            # This difference is purely a model discrepancy, independent of mesh resolution.
            errors = abs(exact-lumped)/abs(exact)
            orders = np.log2(previous/errors) if previous is not None else None
            passed = np.isfinite(errors).all() and (orders is None or min(orders) > .9)
            limits.append(dict(left_resistance_fraction=fraction, thickness_m=thickness,
                           relative_mean_temperature_error=errors[0], relative_outward_flux_errors=errors[1:].tolist(),
                           orders=None if orders is None else orders.tolist(), passed=bool(passed)))
            previous = errors
    print(json.dumps(dict(spatial_runs=runs, thickness_runs=limits, heater_k_W_mK=heater_k,
              sources=[provenance(f) for f in (__file__, getfile(sheet_reference), energy.__file__, thermal.__file__)]),
              indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in runs+limits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
