"""Compare an ambient-driven material interface to its exact harmonic heat flux."""
import argparse
import cmath
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def fem_contact(conductivity, capacity, loss, ambient, omega, resistance, walls, interface, films=None, wall_capacity=(0., 0.)):
    from tensorfem import thermal

    n = len(conductivity)
    x = torch.linspace(0., 1., n+1, dtype=torch.float64)
    start = torch.arange(n)
    contact = {}
    if resistance:
        x = torch.cat((x[:interface+1], x[interface:]))
        start[interface:] += 1
        contact = dict(contact_pairs=torch.tensor([[interface, interface+1]]), contact_conductance=1/resistance)
    fixed = [node for node, wall in zip((0, len(x)-1), walls) if wall is not None]
    prescribed = [wall for wall in walls if wall is not None]
    if films is not None:
        fixed, prescribed = [], []
        end = len(x)-1
        contact.update(convection_edges=torch.tensor([[0, 0], [end, end]]),
                       convection_coefficient=torch.tensor(films, dtype=x.dtype),
                       wall_areal_heat_capacity=torch.tensor(wall_capacity, dtype=x.dtype),
                       ambient_temperature=torch.tensor([w or 0. for w in walls], dtype=torch.complex128))
    model = thermal.ThermalModel(
        x[:, None], torch.stack((start, start+1), dim=1), conductivity, 1., capacity,
        torch.tensor(fixed, dtype=torch.long), torch.tensor(prescribed, dtype=torch.complex128),
        heat_source=loss*ambient, heat_loss_coefficient=loss, **contact)
    if omega == 0:
        state = thermal.solve_low_frequency_thermal(model).steady
    else:
        state = thermal.solve_harmonic_thermal(model, omega)
    field = state.temperature
    inward = thermal.thermal_element_nodal_power(model, field, omega)
    left_flux = -complex(inward[interface-1, 1])
    right_flux = complex(inward[interface, 0])
    wall_state = (field[[0,-1]], torch.stack((inward[0,0], inward[-1,1])))
    return complex(field[start[interface-1]+1]), complex(field[start[interface]]), left_flux, right_flux, wall_state, field[model.elements], inward


def measure(n, omega, resistance, walls, backend, compare_fvm=False, compare_fem=False, left_thickness=.5, films=None, wall_capacity=(0., 0.)):
    k = (1., 2.5)
    capacity = (1.3, .7)
    loss = (2., 4.)
    ambient = (.8-.1j, -.2+.55j)
    interface = round(n*left_thickness)
    if interface/n != left_thickness or not 0 < interface < n:
        raise ValueError("material interface must coincide with an interior mesh face")
    lengths = (left_thickness, 1-left_thickness)
    # Solve k*T''=(H+i*w*C)*T-H*Ta in each layer. At the interface
    # the left and right traces are free[0]-Z[0]*q and free[1]+Z[1]*q.
    free, impedance = [], []
    for side, (conductivity, cv, h, ta, wall, length) in enumerate(zip(k, capacity, loss, ambient, walls, lengths)):
        shift = h + 1j*omega*cv
        wave = cmath.sqrt(shift/conductivity)
        particular = h*ta/shift
        z = cmath.tanh(length*wave)
        if films is not None:
            film = films[side]
            admittance = conductivity*wave
            wall_admittance = film + 1j*omega*wall_capacity[side]
            denominator = wall_admittance + admittance*z
            free.append(particular + (film*(wall or 0.)-wall_admittance*particular)/(cmath.cosh(length*wave)*denominator))
            impedance.append((1 + wall_admittance*z/admittance)/denominator)
        elif wall is None:
            free.append(particular)
            impedance.append(1/(conductivity*wave*z))
        else:
            free.append(particular+(wall-particular)/cmath.cosh(length*wave))
            impedance.append(z/(conductivity*wave))
    exact_flux = (free[0]-free[1])/(resistance+sum(impedance))
    exact_left = free[0]-impedance[0]*exact_flux
    exact_right = free[1]+impedance[1]*exact_flux

    # Propagate the interface state to each wall in its outward coordinate.
    # The reported boundary flux is positive inward at both ends.
    exact_walls = []
    exact_means = []
    centers, cell_means, element_nodes, face_fluxes = [], [], [], []
    for side, trace in enumerate((exact_left, exact_right)):
        shift = loss[side] + 1j*omega*capacity[side]
        particular = loss[side]*ambient[side]/shift
        wave = cmath.sqrt(shift/k[side])
        co, si = cmath.cosh(lengths[side]*wave), cmath.sinh(lengths[side]*wave)
        outward = -exact_flux if side == 0 else exact_flux
        wall_t = particular + co*(trace-particular) - si*outward/(k[side]*wave)
        inward = k[side]*wave*si*(trace-particular) - co*outward
        exact_walls.append((wall_t, inward))
        # Integrate the continuum profile from the contact to the outer wall.
        u = wave*lengths[side]
        exact_means.append(particular + (trace-particular)*si/u - outward*(co-1)/(k[side]*wave*u))

        count = interface if side == 0 else n-interface
        edges = torch.arange(count+1, dtype=torch.float64)/n
        if side == 0:
            edges = edges.flip((0,))
        midpoints = (edges[:-1]+edges[1:])/2

        def excess(distance):
            return (trace-particular)*torch.cosh(wave*distance) - outward*torch.sinh(wave*distance)/(k[side]*wave)

        at_center = excess(midpoints)
        centers.append(particular + at_center)
        # Integrating either exponential over the cell multiplies its midpoint
        # value by sinh(w*dx/2)/(w*dx/2). The constant particular term is exact.
        half = wave/(2*n)
        cell_means.append(particular + at_center*cmath.sinh(half)/half)
        # Fourier flux in +x; the local coordinate points toward the outer wall.
        direction = -1 if side == 0 else 1
        edge_flux = direction*(outward*torch.cosh(wave*edges)
                               - k[side]*wave*(trace-particular)*torch.sinh(wave*edges))
        face_fluxes.append(edge_flux)
        nodes = particular + excess(edges)
        element_nodes.append(torch.stack((nodes[:-1], nodes[1:]), dim=1))

    exact_cuts = torch.cat((face_fluxes[0], face_fluxes[1][1:]))

    def layers(values, dtype=torch.float64):
        return torch.tensor(values, dtype=dtype).repeat_interleave(torch.tensor([interface, n-interface]))

    contact = torch.zeros(n-1, dtype=torch.float64)
    i = interface-1
    contact[i] = resistance
    conductivity = layers(k)
    options = dict(conductivity=conductivity,
        volumetric_heat_capacity=layers(capacity), contact_resistance_x=contact,
        heat_loss_coefficient=layers(loss),
        return_x_flux=True, backend=backend)
    bulk_ambient = layers(ambient, torch.complex128)
    if films is None:
        field, cuts = heat.solve_harmonic_layered_slab(torch.zeros(n), 1/n, walls, omega,
                                                        ambient_temperature=bulk_ambient, **options)
    else:
        field, sph_wall_t, _, cuts = heat.solve_harmonic_layered_films(
            layers(loss)*bulk_ambient, 1/n, tuple(w or 0. for w in walls), omega,
            film_coefficients=films, wall_areal_heat_capacity=wall_capacity, **options)
    left, right = heat.layered_slab_contact_temperatures(field, cuts, 1/n, conductivity)
    flux = complex(cuts[i+1])
    comparison = {"exact_layer_mean_K": [[t.real, t.imag] for t in exact_means]}

    comparison["sph_face_flux_error_W_m2"] = float(abs(cuts-exact_cuts).max())
    comparison["sph_particle_error_K"] = float(abs(field-torch.cat(centers)).max())

    def layer_diagnostics(cell_means, contact_flux, inward):
        means = (complex(cell_means[:interface].mean()), complex(cell_means[interface:].mean()))
        errors = [abs(t-ref) for t, ref in zip(means, exact_means)]
        # Per unit wall area: boundary input plus bulk exchange equals storage.
        boundary_input = (complex(inward[0])-contact_flux, complex(inward[1])+contact_flux)
        residuals = [abs(q + length*(h*(ta-t)-1j*omega*cv*t))
                     for q, length, h, ta, cv, t in zip(boundary_input, lengths, loss, ambient, capacity, means)]
        return dict(mean_temperature_K=[[t.real, t.imag] for t in means],
                    mean_error_K=errors, balance_residual_W_m2=residuals)

    comparison["sph_layers"] = layer_diagnostics(field, flux, (cuts[0], -cuts[-1]))
    def wall_balance(temperatures, inward):
        return max(abs(h*((ta or 0.)-complex(t)) - 1j*omega*c*complex(t) - complex(q))
                   for h, ta, t, c, q in zip(films, walls, temperatures, wall_capacity, inward))

    def wall_errors(temperatures, inward):
        rows = []
        for side, (t, q, (ref_t, ref_q)) in enumerate(zip(temperatures, inward, exact_walls)):
            # A wall with no film or active storage has exactly zero flux.
            has_flux = films[side] != 0 or omega*wall_capacity[side] != 0
            rows.append(dict(temperature_error_K=abs(complex(t)-ref_t),
                             temperature_phase_error_rad=abs(cmath.phase(complex(t)/ref_t)),
                             inward_flux_error_W_m2=abs(complex(q)-ref_q),
                             inward_flux_phase_error_rad=abs(cmath.phase(complex(q)/ref_q)) if has_flux else None))
        return rows

    if films is not None:
        comparison["sph_wall_errors"] = wall_errors(sph_wall_t, (cuts[0], -cuts[-1]))
    if any(wall_capacity):
        comparison["sph_wall_balance_W_m2"] = wall_balance(sph_wall_t, (cuts[0], -cuts[-1]))
    if compare_fvm:
        import numpy as np
        from tensorfvm.cht import rod
        # Unit cross-section converts the rod's power in W to the slab flux in W/m².
        cells, faces, entering, leaving, traces = rod.solve_harmonic_stepped_rod(
            np.full(n, 1/n), np.ones(n), conductivity.numpy(), layers(capacity).numpy(),
            tuple(0. if w is None else w for w in walls), omega,
            end_types=("robin", "robin") if films is not None else tuple("power" if w is None else "temperature" for w in walls),
            end_areal_heat_capacity=wall_capacity, end_film_coefficient=films if films is not None else 0.,
            heat_loss_coefficient=layers(loss).numpy(), bulk_ambient_temperature=layers(ambient, torch.complex128).numpy(),
            contact_resistance=contact.numpy(), return_traces=True)
        comparison["fvm_face_flux_error_W_m2"] = max(float(abs(torch.from_numpy(entering)-exact_cuts[:-1]).max()),
                                                     float(abs(torch.from_numpy(leaving)-exact_cuts[1:]).max()))
        comparison["fvm_cell_mean_error_K"] = float(abs(torch.from_numpy(cells)-torch.cat(cell_means)).max())
        q = leaving[i]
        comparison["fvm_layers"] = layer_diagnostics(cells, complex(q), (entering[0], -leaving[-1]))
        tl, tr = traces[0][i], traces[1][i]
        comparison.update(fvm_flux_relative_error=float(abs(q/exact_flux-1)),
                          fvm_max_trace_error_K=float(max(abs(tl-exact_left), abs(tr-exact_right))),
                          fvm_interface_imbalance_W_m2=float(abs(q-entering[i+1])),
                          solver_flux_difference_W_m2=float(abs(q-flux)),
                          solver_trace_difference_K=float(max(abs(tl-complex(left[i])), abs(tr-complex(right[i])))))
        if films is not None:
            comparison["fvm_wall_errors"] = wall_errors(faces[[0,-1]], (entering[0], -leaving[-1]))
        if any(wall_capacity):
            comparison["fvm_wall_balance_W_m2"] = wall_balance(faces[[0,-1]], (entering[0], -leaving[-1]))
    if compare_fem:
        fl, fr, ql, qr, fem_wall_state, nodal_field, nodal_power = fem_contact(conductivity, layers(capacity), layers(loss),
                                     layers(ambient, torch.complex128), omega, resistance, walls, interface, films, wall_capacity)
        comparison["fem_layers"] = layer_diagnostics(nodal_field.mean(1), ql, fem_wall_state[1])
        comparison["fem_face_flux_error_W_m2"] = max(float(abs(nodal_power[:, 0]-exact_cuts[:-1]).max()),
                                                     float(abs(-nodal_power[:, 1]-exact_cuts[1:]).max()))
        comparison["fem_nodal_error_K"] = float(abs(nodal_field-torch.cat(element_nodes)).max())
        comparison.update(fem_flux_relative_error=abs(ql/exact_flux-1),
                          fem_max_trace_error_K=max(abs(fl-exact_left), abs(fr-exact_right)),
                          fem_interface_imbalance_W_m2=abs(ql-qr),
                          fem_jump_error_K=abs(fl-fr-resistance*exact_flux),
                          fem_sph_flux_difference_W_m2=abs(ql-flux))
        if films is not None:
            comparison["fem_wall_errors"] = wall_errors(*fem_wall_state)
        if any(wall_capacity):
            comparison["fem_wall_balance_W_m2"] = wall_balance(*fem_wall_state)
        if compare_fvm:
            comparison["fem_fvm_flux_difference_W_m2"] = float(abs(ql-q))
    if any(wall_capacity):
        comparison["wall_areal_heat_capacity_J_m2_K"] = wall_capacity
    if films is not None:
        comparison["exact_wall_temperature_K"] = [[t.real, t.imag] for t, q in exact_walls]
        comparison["exact_inward_wall_flux_W_m2"] = [[q.real, q.imag] for t, q in exact_walls]
        comparison.update(wall_film_W_m2_K=films, wall_reservoir_phasors_K=[
            [complex(w or 0.).real, complex(w or 0.).imag] for w in walls])
    return dict(**comparison, columns=n, left_thickness_m=left_thickness, omega_rad_s=omega, contact_resistance_m2_K_W=resistance, backend=backend,
                wall_conditions=["insulated" if w is None else ("film" if films is not None else "zero_temperature_phasor") for w in walls],
                exact_flux_W_m2=[exact_flux.real, exact_flux.imag], flux_relative_error=abs(flux/exact_flux-1),
                flux_phase_error_rad=abs(cmath.phase(flux/exact_flux)),
                max_trace_error_K=max(abs(complex(left[i])-exact_left), abs(complex(right[i])-exact_right)),
                jump_error_K=abs(complex(left[i]-right[i])-resistance*exact_flux))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-fvm", action="store_true", help="also verify the unit-area finite-volume rod")
    parser.add_argument("--compare-fem", action="store_true", help="also verify the unit-area linear finite elements")
    parser.add_argument("--left-thickness", type=float, choices=(.25, .5, .75), default=.5,
                        help="left material thickness in the unit-length slab (m)")
    parser.add_argument("--films", action="store_true", help="drive one or both walls through finite surface films")
    parser.add_argument("--wall-storage", action="store_true", help="include west and east wall storage with the film boundaries")
    args = parser.parse_args()
    files = [__file__, heat.__file__]
    path = "../TensorSPH/src"
    metrics = ["flux_relative_error", "max_trace_error_K", "sph_particle_error_K", "sph_face_flux_error_W_m2"]
    if args.compare_fvm:
        from tensorfvm.cht import rod
        files.append(rod.__file__)
        path += ":../TensorFVM/src"
        metrics.extend(["fvm_flux_relative_error", "fvm_max_trace_error_K", "fvm_cell_mean_error_K", "fvm_face_flux_error_W_m2"])
    if args.compare_fem:
        from tensorfem import thermal
        files.append(thermal.__file__)
        path += ":../TensorFEM/src"
        metrics.extend(["fem_flux_relative_error", "fem_max_trace_error_K", "fem_nodal_error_K", "fem_face_flux_error_W_m2"])
    boundaries = [((0., 0.), None), ((None, 0.), None), ((None, None), None)]
    if args.films or args.wall_storage:
        west, east = .35+.5j, -.4+.15j
        boundaries = [((west, east), (3., 7.)), ((west, None), (3., 0.)), ((None, east), (0., 7.))]
    runs = []
    for backend in ("torch", "banded"):
        for walls, films in boundaries:
            for omega in (0., 12.):
                for resistance in (0., .25):
                    previous = None
                    for n in (16, 32, 64):
                        row = measure(n, omega, resistance, walls, backend, args.compare_fvm, args.compare_fem, args.left_thickness, films,
                                      (.35, .8) if args.wall_storage else (0., 0.))
                        row["passed"] = True
                        for metric in metrics:
                            if previous:
                                order = math.log2(previous[metric]/row[metric])
                                row[metric + "_order"] = order
                                row["passed"] &= order > 1.7
                        if n == 64:
                            row["passed"] &= (row["flux_relative_error"] < .005 and row["max_trace_error_K"] < .001
                                              and row["jump_error_K"] < .001 and row["flux_phase_error_rad"] < .005)
                        if args.compare_fvm:
                            row["passed"] &= row["fvm_interface_imbalance_W_m2"] < 1e-10
                            if n == 64:
                                row["passed"] &= row["fvm_flux_relative_error"] < .005 and row["fvm_max_trace_error_K"] < .001
                        if args.compare_fem:
                            row["passed"] &= row["fem_interface_imbalance_W_m2"] < 1e-10
                            if n == 64:
                                row["passed"] &= (row["fem_flux_relative_error"] < .005 and row["fem_max_trace_error_K"] < .001
                                                  and row["fem_jump_error_K"] < .001)
                        if films is not None and n == 64:
                            for key, wall_rows in row.items():
                                if key.endswith("_wall_errors"):
                                    for wall in wall_rows:
                                        row["passed"] &= (wall["temperature_error_K"] < .001 and wall["inward_flux_error_W_m2"] < .005
                                                          and wall["temperature_phase_error_rad"] < .005
                                                          and (wall["inward_flux_phase_error_rad"] is None
                                                               or wall["inward_flux_phase_error_rad"] < .005))
                        if args.wall_storage:
                            row["passed"] &= all(value < 1e-10 for key, value in row.items() if key.endswith("_wall_balance_W_m2"))
                        for key in ("sph_layers", "fvm_layers", "fem_layers"):
                            if key in row:
                                row["passed"] &= max(row[key]["balance_residual_W_m2"]) < 1e-10
                                error = max(row[key]["mean_error_K"])
                                if previous:
                                    order = math.log2(max(previous[key]["mean_error_K"])/error)
                                    row[key]["mean_error_order"] = order
                                    row["passed"] &= order > 1.7
                                if n == 64:
                                    row["passed"] &= error < .001
                        if n == 64:
                            for metric in ("sph_particle_error_K", "fem_nodal_error_K", "fvm_cell_mean_error_K"):
                                if metric in row:
                                    row["passed"] &= row[metric] < .001
                        if n == 64:
                            row["passed"] &= all(row[k] < .005 for k in metrics if k.endswith("_face_flux_error_W_m2"))
                        runs.append(row)
                        previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(f) for f in files],
                          reproduction=f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH={path} python examples/sph_ambient_driven_contact.py"
                          + (" --compare-fvm" if args.compare_fvm else "") + (" --compare-fem" if args.compare_fem else "")
                          + f" --left-thickness {args.left_thickness}" + (" --films" if args.films else "")
                          + (" --wall-storage" if args.wall_storage else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
