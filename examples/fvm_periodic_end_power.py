"""Time convergence of temperature and cumulative end heat in a periodically driven rod."""
import argparse
import json
import math

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance


def measure(steps, method, alternating=False, compare_fem=False, compare_sph=False, massless=False, single_store=None, bulk_exchange=False):
    # T-base = 0.3*x + (0.7-1.4*x)*sin(omega*t), so the two wall
    # temperatures oscillate in opposite phases and the axial flux is uniform.
    n, area, k, cv = 12, 1.5, 2., 1.2
    omega, base = 2*math.pi, 305.
    ends = np.array([0., 1.])
    centers = (np.arange(n)+.5)/n
    film = np.array([2., 4.])
    capacity = np.zeros(2) if massless else np.array([.25, .6])
    if single_store is not None:
        capacity[1 if single_store == "west" else 0] = 0.
    ambient = np.array([.3, -.2])
    times = np.linspace(0., 1., steps+1)

    if alternating:
        if steps % 2:
            raise ValueError("alternating steps require an even interval count")
        # Short-long pairs have the same end times as pairs of uniform steps.
        times[1::2] -= .5/steps
    dt = np.diff(times)

    def amplitude(x):
        return .7-1.4*x

    loss = 3. if bulk_exchange else 0.
    def reservoir(t, x):
        # cv*T_t = H*(Tb-T) for the affine temperature profile (T_xx=0).
        return .3*x + amplitude(x)*(math.sin(omega*t) + cv*omega/loss*math.cos(omega*t))

    def volume_source(t):
        return 0. if bulk_exchange else cv*omega*amplitude(centers)*math.cos(omega*t)

    def applied(t):
        wave = math.sin(omega*t)
        wall = .3*ends + amplitude(ends)*wave
        q = -k*(.3-1.4*wave)
        return area*(q*np.array([1., -1.]) + capacity*omega*amplitude(ends)*math.cos(omega*t)
                     - film*(ambient-wall))

    field, faces, left, right, heats, means = rod.solve_transient_stepped_rod(
        np.full(n, 1/n), np.full(n, area), k, cv, base+ambient, times, base+.3*centers,
        source=volume_source,
        heat_loss_coefficient=loss, bulk_ambient_temperature=(lambda t: base+reservoir(t, centers)) if bulk_exchange else 0.,
        end_types=("robin", "robin"), end_film_coefficient=film, end_areal_heat_capacity=capacity,
        initial_end_temperature=base+.3*ends, temperature_reference=base, method=method,
        end_applied_power=applied, return_step_heat=True, return_interval_temperatures=True)
    wave = np.sin(omega*times)[:, None]
    exact_t = .3*centers + amplitude(centers)*wave
    exact_wall = .3*ends + amplitude(ends)*wave
    q = -area*k*(.3-1.4*wave)
    # Integrate the analytic conduction power and add wall stored energy
    # since t=0 to obtain the external input at each end at every output time.
    axial_heat = -area*k*(.3*times - 1.4*(1-np.cos(omega*times))/omega)
    exact_heat = axial_heat[:, None]*np.array([1., -1.]) + area*capacity*amplitude(ends)*wave
    heat_error = np.abs(np.cumsum(heats["end_input"], axis=0)-exact_heat[1:]).max(axis=0)
    # Divide the heat error by each interval's actual duration to check
    # average power on short and long intervals at the same physical scale.
    power_error = abs(heats["end_input"]-np.diff(exact_heat, axis=0))/dt[:, None]
    mean_wave = -np.diff(np.cos(omega*times))/(omega*dt)
    mean_wall = .3*ends + amplitude(ends)*mean_wave[:, None]
    film_heat = area*film*(ambient-mean_wall)*dt[:, None]
    applied_heat = np.diff(exact_heat, axis=0)-film_heat
    split_residual = heats["end_input"]-heats["end_applied"]-heats["end_film"]
    wall_energy = area*capacity*np.diff(faces[:, [0,-1]], axis=0)
    conduction = np.column_stack((heats["left"][:, 0], -heats["right"][:, -1]))
    residual = heats["end_input"]-conduction-wall_energy
    # Independent wall storage from external input less heat conducted into
    # the rod, compared with the exact change in wall energy at each end.
    exact_storage = area*capacity*amplitude(ends)*np.diff(wave, axis=0)
    def storage_errors(external_heat, conductive_heat):
        return (abs(external_heat-conductive_heat-exact_storage)/dt[:, None]).max(axis=0).tolist()

    storage = {"fvm": storage_errors(heats["end_input"], conduction)}
    exact_bulk = area*cv/n*amplitude(centers)*np.diff(wave, axis=0)
    def bulk_power_error(cell_heat):
        return float((abs(cell_heat-exact_bulk)/dt[:, None]).max())
    bulk_errors = {}
    if bulk_exchange:
        bulk_errors["fvm_bulk_power_error_W"] = bulk_power_error(heats["bulk_exchange"])
    fem_errors = {}
    if compare_fem:
        import torch
        from tensorfem import thermal

        # Solve for excess temperature relative to base, with identical physical loads.
        x = torch.linspace(0., 1., n+1, dtype=torch.float64)
        elements = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
        model = thermal.ThermalModel(x[:, None], elements, k, 1., cv, elements.new_empty(0), x[:0],
            thickness=area, heat_loss_coefficient=loss, convection_edges=elements.new_tensor([[0,0], [n,n]]),
            convection_area=x.new_full((2,), area), convection_coefficient=torch.from_numpy(film),
            ambient_temperature=torch.from_numpy(ambient), wall_areal_heat_capacity=torch.from_numpy(capacity))
        # Consistent LINE2 projection of the spatially affine volume source.
        projection = area*cv*omega/(6*n)*(amplitude(x)[elements] @ x.new_tensor([[2.,1.],[1.,2.]]))
        volume_power = x.new_zeros(n+1)
        volume_power[:-1] += projection[:, 0]
        volume_power[1:] += projection[:, 1]

        if bulk_exchange:
            mass = area/(6*n)*x.new_tensor([[2.,1.],[1.,2.]])
            def reservoir_load(t):
                return loss*reservoir(t, x)[elements] @ mass

        def nodal_power(t):
            load = volume_power*math.cos(omega*t)
            if bulk_exchange:
                element_load = reservoir_load(t)
                load = x.new_zeros(n+1)
                load[:-1] += element_load[:, 0]
                load[1:] += element_load[:, 1]
            load[[0,-1]] += torch.from_numpy(applied(float(t)))
            return load

        result = thermal.solve_transient_thermal(model, torch.from_numpy(times), .3*x, theta=.5,
                                                 method=method, nodal_heat_history=nodal_power)
        gamma = 1.-1./math.sqrt(2.) if method == "sdirk2" else .5
        first = times[:-1]+gamma*dt if method == "sdirk2" else times[:-1]
        cosine = (1-gamma)*np.cos(omega*first)+gamma*np.cos(omega*times[1:])
        distributed = torch.from_numpy(dt*cosine)[:, None, None]*projection
        if bulk_exchange:
            stage_load = torch.stack([(1-gamma)*reservoir_load(a)+gamma*reservoir_load(b)
                                      for a, b in zip(first, times[1:])])
            distributed = torch.from_numpy(dt)[:, None, None]*stage_load
            fem_means = result.interval_temperature[:, elements].mean(2).numpy()
            bulk_errors["fem_bulk_power_error_W"] = bulk_power_error(
                distributed.sum(2).numpy() - dt[:, None]*loss*area/n*fem_means)
        local = thermal.thermal_element_nodal_heat_steps(model, result, element_nodal_step_heat=distributed)
        face_heat = torch.stack((local[:,0,0], local[:,-1,1]), dim=1).numpy()
        wall_t = result.temperature[:, [0,-1]].numpy()
        recovered = face_heat + area*capacity*np.diff(wall_t, axis=0)
        film_fem = area*film*(ambient-result.interval_temperature[:, [0,-1]].numpy())*dt[:, None]
        heater = (result.additional_stage_power[:, [0,-1]].numpy()
                  - torch.stack((distributed[:,0,0], distributed[:,-1,1]), dim=1).numpy()/dt[:, None])*dt[:, None]
        storage["fem"] = storage_errors(heater+film_fem, face_heat)
        fem_errors = dict(fem_wall_error_K=float(abs(wall_t-exact_wall).max()),
                          fem_cumulative_end_error_J=float(abs(recovered.cumsum(axis=0)-exact_heat[1:]).max()),
                          fem_interval_film_error_W=float((abs(film_fem-film_heat)/dt[:, None]).max()),
                          fem_interval_heater_error_W=float((abs(heater-applied_heat)/dt[:, None]).max()),
                          fem_wall_balance_residual_J=float(abs(recovered-heater-film_fem).max()),
                          fem_fvm_cumulative_difference_J=float(abs((recovered-heats["end_input"]).cumsum(axis=0)).max()))
    sph_errors = {}
    if compare_sph:
        from tensorsph import heat_conduction_2d as sph

        # The slab integrates per unit area. Convert each end input to W/m²
        # and the returned heats back to J before comparing to the rod.
        wall_loads = tuple(lambda t, i=i: applied(t)[i]/area for i in range(2))
        state, cuts, wall, bulk, external, wall_mean, components = sph.solve_transient_layered_slab(
            .3*centers, times, 1/n, (None, None), conductivity=np.full(n, k),
            volumetric_heat_capacity=cv, heat_source=volume_source, heat_loss_coefficient=loss,
            bulk_ambient_temperature=(lambda t: reservoir(t, centers)) if bulk_exchange else 0., return_bulk_exchange=True,
            x_wall_films=tuple(zip(film, ambient)), x_wall_inward_flux=wall_loads,
            wall_areal_heat_capacity=capacity, initial_wall_temperatures=tuple(.3*x if c else None for x, c in zip(ends, capacity)),
            method=method, return_x_heat=True, return_external_heat=True,
            return_wall_temperatures=True, return_interval_wall_temperatures=True, return_wall_heat_components=True)
        if bulk_exchange:
            bulk_errors["sph_bulk_power_error_W"] = bulk_power_error(area*bulk.numpy())
        external = area*external.numpy()
        film_sph = area*components["film"].numpy()
        heater_sph = area*components["applied"].numpy()
        conduction_sph = area*cuts.numpy()[:, [0,-1]]*np.array([1., -1.])
        storage_sph = area*capacity*np.diff(wall.numpy(), axis=0)
        storage["sph"] = storage_errors(external, conduction_sph)
        sph_errors = dict(sph_particle_error_K=float(abs(state.numpy()-exact_t).max()),
                          sph_wall_error_K=float(abs(wall.numpy()-exact_wall).max()),
                          sph_cumulative_end_error_J=float(abs(external.cumsum(axis=0)-exact_heat[1:]).max()),
                          sph_interval_film_error_W=float((abs(film_sph-film_heat)/dt[:, None]).max()),
                          sph_interval_heater_error_W=float((abs(heater_sph-applied_heat)/dt[:, None]).max()),
                          sph_component_sum_residual_J=float(abs(external-film_sph-heater_sph).max()),
                          sph_film_law_residual_J=float(abs(film_sph-area*film*(ambient-wall_mean.numpy())*dt[:, None]).max()),
                          sph_wall_balance_residual_J=float(abs(external-conduction_sph-storage_sph).max()))
    return dict(**bulk_errors, wall_capacity_J_m2_K=capacity.tolist(), storage_power_errors_W=storage, **sph_errors, **fem_errors, min_dt_s=float(dt.min()), max_dt_s=float(dt.max()),
                interval_heater_power_error_W=float((abs(heats["end_applied"]-applied_heat)/dt[:, None]).max()),
                interval_film_power_error_W=float((abs(heats["end_film"]-film_heat)/dt[:, None]).max()),
                max_input_split_residual_J=float(abs(split_residual).max()),
                max_wall_energy_residual_J=float(abs(residual).max()),
                interval_end_power_error_W=float(power_error.max()),
                interval_wall_mean_error_K=float(abs(means["face"][:, [0,-1]]-mean_wall).max()), steps=steps, method=method, columns=n,
                max_cell_error_K=float(abs(field-exact_t).max()),
                max_wall_error_K=float(abs(faces[:, [0,-1]]-exact_wall).max()),
                max_section_error_W=float(max(abs(left-q).max(), abs(right-q).max())),
                west_cumulative_error_J=float(heat_error[0]), east_cumulative_error_J=float(heat_error[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alternating", action="store_true", help="alternate short and long steps in a 1:3 ratio")
    parser.add_argument("--compare-fem", action="store_true", help="recover end heat from linear finite elements")
    parser.add_argument("--compare-sph", action="store_true", help="verify the same external heat on a layered particle slab")
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument("--single-store", choices=("west", "east"), help="retain surface heat capacity only at the selected end")
    storage_group.add_argument("--massless", action="store_true", help="apply heaters and convection at walls without surface storage")
    parser.add_argument("--bulk-exchange", action="store_true", help="drive the interior with a space- and time-dependent reservoir")
    args = parser.parse_args()
    files = [__file__, rod.__file__]
    path = "../TensorFVM/src"
    fem_metrics = ()
    if args.compare_fem:
        from tensorfem import thermal
        files.append(thermal.__file__)
        path += ":../TensorFEM/src"
        fem_metrics = ("fem_wall_error_K", "fem_cumulative_end_error_J",
                       "fem_interval_film_error_W", "fem_interval_heater_error_W")
    sph_metrics = ()
    if args.compare_sph:
        from tensorsph import heat_conduction_2d as sph
        files.append(sph.__file__)
        path += ":../TensorSPH/src"
        sph_metrics = ("sph_particle_error_K", "sph_wall_error_K", "sph_cumulative_end_error_J",
                       "sph_interval_film_error_W", "sph_interval_heater_error_W")
    bulk_metrics = [name+"_bulk_power_error_W" for name, active in (("fvm", True), ("fem", args.compare_fem), ("sph", args.compare_sph)) if args.bulk_exchange and active]
    runs = []
    for method in ("theta", "sdirk2"):
        previous = None
        for steps in (32, 64, 128):
            row = measure(steps, method, args.alternating, args.compare_fem, args.compare_sph, args.massless, args.single_store, args.bulk_exchange)
            row["passed"] = max(row["max_wall_energy_residual_J"], row["max_input_split_residual_J"]) < 1e-11
            for key in ("max_cell_error_K", "max_wall_error_K", "max_section_error_W",
                        "west_cumulative_error_J", "east_cumulative_error_J",
                        "interval_end_power_error_W", "interval_wall_mean_error_K",
                        "interval_heater_power_error_W", "interval_film_power_error_W", *fem_metrics, *sph_metrics):
                if previous:
                    order = math.log2(previous[key]/row[key])
                    row[key+"_order"] = order
                    row["passed"] &= order > 1.8
            if steps == 128:
                row["passed"] &= max(row["west_cumulative_error_J"], row["east_cumulative_error_J"]) < .002
            if args.compare_fem:
                row["passed"] &= row["fem_wall_balance_residual_J"] < 1e-11
                if steps == 128:
                    row["passed"] &= row["fem_cumulative_end_error_J"] < .002
            if args.compare_sph:
                row["passed"] &= max(row["sph_wall_balance_residual_J"], row["sph_component_sum_residual_J"],
                                     row["sph_film_law_residual_J"]) < 1e-11
                if steps == 128:
                    row["passed"] &= row["sph_cumulative_end_error_J"] < .002
            for errors in row["storage_power_errors_W"].values():
                for capacity, error in zip(row["wall_capacity_J_m2_K"], errors):
                    if capacity == 0:
                        row["passed"] &= error < 1e-10
                    elif steps == 128:
                        row["passed"] &= error < .005
            for key in bulk_metrics:
                if previous:
                    row[key+"_order"] = math.log2(previous[key]/row[key])
                    row["passed"] &= row[key+"_order"] > 1.8
                if steps == 128:
                    row["passed"] &= row[key] < .001
            runs.append(row)
            previous = row
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(f) for f in files],
                          reproduction=f"OPENBLAS_NUM_THREADS=1 PYTHONPATH={path} python examples/fvm_periodic_end_power.py"
                          + (" --alternating" if args.alternating else "") + (" --compare-fem" if args.compare_fem else "")
                          + (" --compare-sph" if args.compare_sph else "") + (" --massless" if args.massless else "")
                          + (f" --single-store {args.single_store}" if args.single_store else "") + (" --bulk-exchange" if args.bulk_exchange else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
