"""Verify FEM and DG material-side heat fluxes for a heated slab with storing end walls."""
import json
from dataclasses import replace

import numpy as np
import torch

from conduction_provenance import provenance
from periodic_contact_heat import reference
from two_layer_conduction import slab_fem_model
from tensordg import heat_1d as dg
from tensorfem import thermal


def measure(n, omega):
    resistance = 0.32
    k, c = (1., 4.), (1., 2.)
    film = np.array([1.8, 3.5])
    wall_capacity = np.array([.37, .81])
    base, amplitude = -.25+.4j, .8-.3j
    nodes, _, contact, inward = reference(2*n, resistance, omega)
    exact = base + amplitude*nodes[[0, n, n+1, -1]]
    inward = amplitude*inward
    contact *= amplitude
    ambient = exact[[0, -1]] + (inward+1j*omega*wall_capacity*exact[[0, -1]])/film
    source = 1j*omega*np.array(c)*base

    matrices, offsets = zip(*(dg.harmonic_heat_wall_admittance_1d(
        omega, ncells=n, degree=2, thickness=.5, conductivity=k[i], capacity=c[i],
        source=source[i]) for i in (0, 1)))
    opts = dict(layer_flux_offsets=offsets)
    y, offset = dg.compose_heat_wall_admittances_1d(matrices, resistance, return_flux_offset=True, **opts)
    wall, q = dg.solve_heat_wall_admittance_1d(
        y, ("robin", "robin"), ambient, flux_offset=offset, film_coefficients=film,
        angular_frequency=omega, wall_areal_heat_capacity=wall_capacity)
    _, traces, flux = dg.compose_heat_wall_admittances_1d(matrices, resistance, wall_temperatures=wall, **opts)
    solutions = [("DG_P2", traces.flatten().numpy(), q.numpy(), -flux[0, 1].item())]

    model, ends = slab_fem_model(np.repeat(k, n), 0., 0., resistance, film,
                                torch.from_numpy(np.repeat(c, n)))
    model = replace(model, ambient_temperature=torch.from_numpy(ambient),
                    heat_source=torch.from_numpy(np.repeat(source, n)),
                    wall_areal_heat_capacity=torch.from_numpy(wall_capacity))
    state = thermal.solve_harmonic_thermal(model, omega, sparse=True).temperature
    power = thermal.thermal_element_nodal_power(model, state, omega)
    # The unit cross section makes the two Q4 endpoint powers sum to the
    # numerical value of the inward flux density on each wall.
    q = torch.stack((power[0, [0, 3]].sum(), power[-1, [1, 2]].sum())).numpy()
    surfaces = state.reshape(2, -1).mean(0)[[0, n, n+1, -1]].numpy()
    solutions.append(("FEM_Q4", surfaces, q, -power[n-1, [1, 2]].sum().item()))

    for solver, t, material, cut in solutions:
        end_t = t[[0, -1]]
        yield dict(solver=solver, cells_per_layer=n, omega_rad_s=omega,
                   trace_error_K=float(max(abs(t-exact))),
                   end_flux_error_W_m2=float(max(abs(material-inward))),
                   interface_flux_error_W_m2=float(abs(cut-contact)),
                   contact_drop_residual_K=float(abs(t[1]-t[2]-resistance*cut)),
                   wall_balance_residual_W_m2=float(max(abs(film*(ambient-end_t)-1j*omega*wall_capacity*end_t-material))))


if __name__ == "__main__":
    runs = []
    for omega in (1., 6.):
        previous = {}
        for n in (4, 8, 16, 32):
            for row in measure(n, omega):
                solver = row["solver"]
                errors = np.array([row[key] for key in ("trace_error_K", "end_flux_error_W_m2",
                                                        "interface_flux_error_W_m2")])
                row["passed"] = bool(max(row["contact_drop_residual_K"], row["wall_balance_residual_W_m2"]) < 1e-10
                                     and (solver not in previous or np.all(errors < previous[solver]/3))
                                     and (n != 32 or max(errors) < 5e-4))
                previous[solver] = errors
                runs.append(row)
    print(json.dumps(dict(runs=runs, sources=[provenance(path) for path in (
        __file__, reference.__code__.co_filename, slab_fem_model.__code__.co_filename,
        dg.__file__, thermal.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
