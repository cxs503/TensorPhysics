"""Verify FEM, DG, FVM, SPH and thermal contact-chain material-side heat fluxes for a heated slab with storing end walls."""
import json
from dataclasses import replace

import numpy as np
import torch

from conduction_provenance import provenance
from periodic_contact_heat import reference
from two_layer_conduction import slab_fem_model
from tensordem.thermal import contact_chain
from tensordg import heat_1d as dg
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as sph


def measure(n, omega):
    resistance = 0.32
    k, c = (1., 4.), (1., 2.)
    film = np.array([1.8, 3.5])
    wall_capacity = np.array([.37, .81])
    base, amplitude = -.25+.4j, .8-.3j
    nodes, mean, contact, inward = reference(2*n, resistance, omega)
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

    dx = np.full(2*n, .5/n)
    contacts = np.zeros(2*n-1)
    contacts[n-1] = resistance
    cell, faces, from_left, to_right, (hot, cold) = rod.solve_harmonic_stepped_rod(
        dx, 1., np.repeat(k, n), np.repeat(c, n), ambient, omega,
        end_types=("robin", "robin"), end_film_coefficient=film, end_areal_heat_capacity=wall_capacity,
        contact_resistance=contacts, source=np.repeat(source, n), flux_order=2, return_traces=True)
    surfaces = np.array([faces[0], hot[n-1], cold[n-1], faces[-1]])
    finite_volume = dict(
        cell_mean_error_K=float(max(abs(cell-(base+amplitude*mean)))),
        cell_balance_residual_W_m2=float(max(abs(from_left-to_right
            + dx*(np.repeat(source, n)-1j*omega*np.repeat(c, n)*cell)))),
        interface_flux_mismatch_W_m2=float(abs(to_right[n-1]-from_left[n])))
    solutions.append(("FVM", surfaces, np.array([from_left[0], -to_right[-1]]), to_right[n-1]))

    spacing = dx[0]
    particles, walls, wall_flux, cuts = sph.solve_harmonic_layered_films(
        np.repeat(source, n), spacing, ambient, omega, film_coefficients=film,
        conductivity=np.repeat(k, n), volumetric_heat_capacity=np.repeat(c, n),
        contact_resistance_x=contacts, wall_areal_heat_capacity=wall_capacity,
        backend="banded", return_x_flux=True)
    left, right = sph.layered_slab_contact_temperatures(particles, cuts, spacing, np.repeat(k, n))
    surfaces = np.array([walls[0], left[n-1], right[n-1], walls[1]])
    # Convert the analytic mean of the homogeneous mode to a midpoint value.
    z = spacing/2*np.sqrt(1j*omega*np.repeat(c, n)/np.repeat(k, n))
    exact_points = base + amplitude*mean*z/np.sinh(z)
    particle_metrics = dict(
        particle_temperature_error_K=float(max(abs(particles.numpy()-exact_points))),
        particle_balance_residual_W_m2=float(max(abs(cuts.diff().numpy()
            + spacing*(1j*omega*np.repeat(c, n)*particles.numpy()-np.repeat(source, n))))),
        cut_wall_flux_mismatch_W_m2=float(max(abs(cuts[[0, -1]].numpy()*[1, -1]-wall_flux.numpy()))))
    solutions.append(("SPH", surfaces, wall_flux.numpy(), cuts[n].item()))
    # Unit area converts the contact chain's link powers to flux densities.
    links, stores = contact_chain.cell_centered_slab_network(
        dx, np.repeat(k, n), np.repeat(c, n), films=film, wall_areal_heat_capacity=wall_capacity,
        contact_resistance=contacts, resolve_contacts=True)
    # Positive-capacity nodes are the left wall, material centers, and right wall.
    centers = np.flatnonzero(stores)[1:-1]
    left_trace, right_trace = centers[n-1]+1, centers[n]-1
    applied = np.zeros(stores.size, dtype=complex)
    applied[centers] = dx*np.repeat(source, n)
    chain = contact_chain.harmonic_contact_temperatures(
        links, stores, omega, t_left=ambient[0], t_right=ambient[1],
        nodal_power=applied, allow_massless=True)
    link_flux = -np.diff(chain)/links
    surfaces = chain[[1, left_trace, right_trace, -2]]
    solutions.append(("DEM_thermal_chain", surfaces, link_flux[[1, -2]]*[1, -1], link_flux[left_trace]))
    chain_metrics = dict(
        center_temperature_error_K=float(max(abs(chain[centers]-exact_points))),
        free_node_balance_residual_W_m2=float(max(abs(np.diff(link_flux)
            + 1j*omega*stores[1:-1]*chain[1:-1]-applied[1:-1]))))
    extra = {"FVM": finite_volume, "SPH": particle_metrics, "DEM_thermal_chain": chain_metrics}

    for solver, t, material, cut in solutions:
        end_t = t[[0, -1]]
        yield dict(**extra.get(solver, {}), solver=solver, cells_per_layer=n, omega_rad_s=omega,
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
                residuals = [row["wall_balance_residual_W_m2"]]
                if solver == "SPH":
                    # SPH recovers each material trace by half-cell extrapolation.
                    errors = np.append(errors, [row["particle_temperature_error_K"], row["contact_drop_residual_K"]])
                    residuals.extend([row["particle_balance_residual_W_m2"], row["cut_wall_flux_mismatch_W_m2"]])
                else:
                    residuals.append(row["contact_drop_residual_K"])
                if solver == "DEM_thermal_chain":
                    errors = np.append(errors, row["center_temperature_error_K"])
                    residuals.append(row["free_node_balance_residual_W_m2"])
                if solver == "FVM":
                    errors = np.append(errors, row["cell_mean_error_K"])
                    residuals.extend([row["cell_balance_residual_W_m2"], row["interface_flux_mismatch_W_m2"]])
                row["passed"] = bool(max(residuals) < 1e-10
                                     and (solver not in previous or np.all(errors < previous[solver]/3))
                                     and (n != 32 or max(errors) < 5e-4))
                previous[solver] = errors
                runs.append(row)
    print(json.dumps(dict(runs=runs, sources=[provenance(path) for path in (
        __file__, reference.__code__.co_filename, slab_fem_model.__code__.co_filename,
        dg.__file__, thermal.__file__, rod.__file__, sph.__file__, contact_chain.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
