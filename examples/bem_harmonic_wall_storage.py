"""Verify harmonic heat conduction in a unit square slab with surface heat storage."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def square_mesh(panels_per_side):
    corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    t = np.arange(panels_per_side)[:, None]/panels_per_side
    return geometry.PanelMesh(np.concatenate([
        a+(b-a)*t for a, b in zip(corners, np.roll(corners, -1, axis=0))]))


def measure(panels_per_side, omega, quadratic):
    k, bulk_capacity, wall_capacity, h = 1.3, 2., .8, 2.4
    ambient = .9-.3j
    mesh = square_mesh(panels_per_side)
    right = mesh.normals[:, 0] == 1
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, np.where(right, "film", "flux"), np.where(right, ambient, 0.), omega,
        conductivity=k, volumetric_heat_capacity=bulk_capacity, film_coefficients=h,
        wall_areal_heat_capacity=right*wall_capacity, quadratic=quadratic)

    # The insulated end imposes T(x)=A cosh(lambda x). At x=1 the
    # film supplies both bulk conduction and the thin wall's storage.
    wave = np.sqrt(1j*omega*bulk_capacity/k)
    amplitude = h*ambient/((h+1j*omega*wall_capacity)*np.cosh(wave)+k*wave*np.sinh(wave))
    wall = amplitude*np.cosh(wave)
    bulk_flux = k*wave*amplitude*np.sinh(wave)
    surface_rate = 1j*omega*wall_capacity*wall
    sample_right = np.repeat(right, 3 if quadratic else 1)
    temp = result.wall_temperature[sample_right]
    flux = result.inward_flux[sample_right]
    external = result.external_inward_flux[sample_right]
    x = np.array([.1, .35, .6, .9])
    probe = result.temperature(np.column_stack((x, np.full_like(x, .47))))
    return dict(panels=mesh.panel_count, omega_rad_s=omega, quadratic=quadratic,
                probe_error_K=float(max(abs(probe-amplitude*np.cosh(wave*x)))),
                wall_error_K=float(max(abs(temp-wall))),
                bulk_flux_error_W_m2=float(max(abs(flux-bulk_flux))),
                surface_storage_error_W_m2=float(max(abs((external-flux)-surface_rate))),
                external_flux_error_W_m2=float(max(abs(external-h*(ambient-wall)))))


def two_walls(n, omega, bulk_exchange=False, reservoir_gradient=0j):
    """Independent slab solution with flux drive on the left, film on the right."""
    k, c, h = 1.3, 2., 2.4
    capacities = np.array([.45, .8])
    loss = 3.5 if bulk_exchange else 0.
    reservoir = .25-.4j
    drive, ambient, heater = -.35+.6j, .9-.3j, .4+.2j
    mesh = square_mesh(n)
    left, right = mesh.normals[:, 0] == -1, mesh.normals[:, 0] == 1
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, np.where(right, "film", "flux"), np.select([left, right], [drive, ambient]),
        omega, conductivity=k, volumetric_heat_capacity=c, film_coefficients=h,
        volumetric_source=heater, wall_areal_heat_capacity=left*capacities[0]+right*capacities[1],
        heat_loss_coefficient=loss, bulk_ambient_temperature=reservoir,
        bulk_ambient_gradient=(reservoir_gradient, 0.), quadratic=True)

    # Write T=P+A cosh(lambda x)+B sinh(lambda x). The outward derivatives
    # at x=0 and x=1 give the two material influxes with opposite signs.
    admittance = loss+1j*omega*c
    wave = np.sqrt(admittance/k)
    p = (heater+loss*reservoir)/admittance
    slope = loss*reservoir_gradient/admittance
    sh, ch = np.sinh(wave), np.cosh(wave)
    yl, yr = 1j*omega*capacities + [0., h]
    a, b = np.linalg.solve([[yl, -k*wave],
                            [k*wave*sh+yr*ch, k*wave*ch+yr*sh]],
                           [drive-yl*p+k*slope, h*ambient-yr*(p+slope)-k*slope])
    exact_wall = np.array([p+a, p+slope+a*ch+b*sh])
    exact_flux = k*wave*np.array([-b, a*sh+b*ch])+k*slope*np.array([-1., 1.])
    exact_mean = p+slope/2+(a*sh+b*(ch-1))/wave
    exact_surface = 1j*omega*capacities*exact_wall
    exact_external = np.array([drive, h*(ambient-exact_wall[1])])

    def end_traces(values):
        return np.array([values[np.repeat(side, 3)] for side in (left, right)])

    wall = end_traces(result.wall_temperature)
    flux = end_traces(result.inward_flux)
    external = end_traces(result.external_inward_flux)
    storage = 1j*omega*capacities[:, None]*wall
    # Three-node Gauss weights integrate each quadratic boundary trace.
    weights = np.tile(np.array([5., 8., 5.])/(18*n), n)
    nodes, gauss = np.polynomial.legendre.leggauss(10)
    x, y = np.meshgrid((1+nodes)/2, (1+nodes)/2)
    targets = np.column_stack((x.ravel(), y.ravel()))
    probe = result.temperature(targets).reshape(x.shape)
    mean = (gauss @ probe @ gauss)/4
    bulk_rate = 1j*omega*c*mean
    distributed_heat = loss*(reservoir+reservoir_gradient/2-mean)

    def trace_error(samples, exact):
        return float(np.max(abs(samples-exact[:, None])))

    row = dict(panels=mesh.panel_count, omega_rad_s=omega,
                # Report signed phasors so that phase and orientation can be inspected.
                end_influx_W_m2=[[z.real, z.imag] for z in flux @ weights],
                exact_end_influx_W_m2=[[z.real, z.imag] for z in exact_flux],
                wall_error_K=trace_error(wall, exact_wall),
                probe_error_K=float(np.max(abs(probe-(p+slope*x+a*np.cosh(wave*x)+b*np.sinh(wave*x))))),
                bulk_flux_error_W_m2=trace_error(flux, exact_flux),
                external_flux_error_W_m2=trace_error(external, exact_external),
                surface_storage_error_W_m2=trace_error(storage, exact_surface),
                bulk_storage_error_W_m=float(abs(bulk_rate-1j*omega*c*exact_mean)),
                energy_residual_W_m=float(abs((external-storage).sum(axis=0) @ weights+heater+distributed_heat-bulk_rate)))
    # Both physical windows cut through each end wall's first and last
    # intersected panels. The slab solution is uniform along each wall.
    panels = np.arange(mesh.panel_count)
    window_errors = dict(bulk=[], storage=[], external=[])
    defects = []
    for side_id, side in enumerate((left, right)):
        for lo, hi in ((.13, .77), (.34, .91)):
            begin = np.clip(n*lo-np.arange(n), 0., 1.)
            end = np.clip(n*hi-np.arange(n), 0., 1.)
            loads = {}
            for component, exact in (("bulk", exact_flux), ("storage", exact_surface), ("external", exact_external)):
                loads[component] = result.panel_inward_power(panels[side], begin, end, component=component).sum()
                window_errors[component].append(abs(loads[component]-(hi-lo)*exact[side_id]))
            defects.append(abs(loads["external"]-loads["bulk"]-loads["storage"]))
    for component, errors in window_errors.items():
        row[f"window_{component}_power_error_W_m"] = float(max(errors))
    row["window_balance_defect_W_m"] = float(max(defects))
    if bulk_exchange:
        row["distributed_heat_error_W_m"] = float(abs(distributed_heat-loss*(reservoir+reservoir_gradient/2-exact_mean)))
        row["distributed_heat_W_m"] = [distributed_heat.real, distributed_heat.imag]
    if reservoir_gradient:
        interior_flux = result.heat_flux(targets)
        exact_x = -k*(slope+wave*(a*np.sinh(wave*targets[:, 0])+b*np.cosh(wave*targets[:, 0])))
        row["interior_flux_error_W_m2"] = float(np.max(abs(interior_flux-np.column_stack((exact_x, np.zeros_like(exact_x))))))
    return row


def verify_two_walls(bulk_exchange=False, reservoir_gradient=0j):
    rows = []
    passed = True
    for omega in (.5, 4.):
        series = [two_walls(n, omega, bulk_exchange, reservoir_gradient) for n in (8, 16, 32)]
        rows.extend(series)
        passed &= all(row["window_balance_defect_W_m"] < 1e-12 for row in series)
        for metric in series[-1]:
            if "_error_" in metric or metric == "energy_residual_W_m":
                errors = [row[metric] for row in series]
                passed &= all(b < a for a, b in zip(errors, errors[1:])) and errors[-1] < 2e-5
    command = "OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_harmonic_wall_storage.py --two-walls"
    if bulk_exchange:
        command += " --affine-reservoir" if reservoir_gradient else " --bulk-exchange"
    print(json.dumps(dict(reference="Heated unit square slab with two storing walls, left flux and right film",
                          reservoir_gradient_x_K_m=[reservoir_gradient.real, reservoir_gradient.imag],
                          bulk_exchange_W_m3_K=3.5 if bulk_exchange else 0.,
                          reproduction=command,
                          solver=provenance(harmonic_conduction.__file__), benchmark=provenance(__file__),
                          runs=rows, passed=passed), indent=2, allow_nan=False))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-walls", action="store_true", help="verify mixed drives and whole-slab energy balance")
    parser.add_argument("--bulk-exchange", action="store_true", help="run the two-wall slab with distributed reservoir exchange")
    parser.add_argument("--affine-reservoir", action="store_true", help="use a spatially varying bulk ambient phasor")
    args = parser.parse_args()
    if args.affine_reservoir:
        raise SystemExit(0 if verify_two_walls(True, .6+.15j) else 1)
    if args.two_walls or args.bulk_exchange:
        raise SystemExit(0 if verify_two_walls(args.bulk_exchange) else 1)
    rows = []
    passed = True
    for quadratic in (False, True):
        for omega in (.5, 4.):
            levels = (8, 16, 32) if quadratic else (8, 16, 32, 64, 128)
            series = [measure(n, omega, quadratic) for n in levels]
            rows.extend(series)
            for metric in series[-1]:
                if "_error_" in metric:
                    errors = [row[metric] for row in series]
                    passed &= all(b < a for a, b in zip(errors, errors[1:])) and errors[-1] < .003
    print(json.dumps(dict(reference="T(x)=A cosh(lambda x), insulated at x=0, storing film at x=1",
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_harmonic_wall_storage.py",
                          solver=provenance(harmonic_conduction.__file__), benchmark=provenance(__file__),
                          runs=rows, passed=passed), indent=2))
    raise SystemExit(0 if passed else 1)
