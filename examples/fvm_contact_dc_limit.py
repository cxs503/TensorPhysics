"""Verify reconstructed harmonic rod fields from DC through finite frequency."""
import json

import numpy as np
from numpy.polynomial import Polynomial
from tensorfvm.cht import rod

import periodic_contact_reference as slab
from conduction_provenance import provenance


def dc_slopes(x, interface, resistance, conductivities, capacities):
    """Differentiate the continuum balance with respect to i*omega at DC."""
    lengths = (interface, 1-interface)
    total_resistance = resistance + sum(h/k for h, k in zip(lengths, conductivities))
    steady_flux = 1/total_resistance
    temperature, lag, flux_lag = 1., 0., 0.
    fields = []
    for layer, (h, k, c) in enumerate(zip(lengths, conductivities, capacities)):
        steady = Polynomial([temperature, -steady_flux/k])
        # q1' = -C*T0 and T1' = -q1/k. Initially guess q1(0)=0.
        q1 = (-c*steady).integ(k=flux_lag)
        t1 = (-q1/k).integ(k=lag)
        fields.append((t1, q1))
        temperature, lag, flux_lag = steady(h), t1(h), q1(h)
        if layer == 0:
            temperature -= resistance*steady_flux
            lag -= resistance*flux_lag
    # Both imposed wall temperatures have zero frequency derivative.
    correction = lag/total_resistance
    x_right = x-interface
    path_resistance = np.where(x < interface, x/conductivities[0],
                               interface/conductivities[0]+resistance+x_right/conductivities[1])
    t1 = np.where(x < interface, fields[0][0](x), fields[1][0](x_right))
    q1 = np.where(x < interface, fields[0][1](x), fields[1][1](x_right))
    return t1-correction*path_resistance, q1+correction


def measure(n, omega):
    cut = 3*n//8
    width = np.full(n, 1/n)
    k = np.where(np.arange(n) < cut, 1., 4.)
    capacity = np.where(np.arange(n) < cut, 1., 2.)
    resistance = .24
    contact = np.zeros(n-1)
    contact[cut-1] = resistance
    mean, _, west, east, (gradient, curvature) = rod.solve_harmonic_stepped_rod(
        width, 1., k, capacity, (1., 0.), omega, contact_resistance=contact,
        flux_order=2, source_derivative_breaks=[cut], return_source_derivatives=True)
    fractions = np.array([.25, .5, .75])
    temperature, power = rod.reconstruct_steady_rod_field(
        width, 1., k, mean, west, east, fractions,
        source_gradient=gradient, source_curvature=curvature)
    nodes, _, _, _, flux = slab.reference(
        4*n, resistance, omega, contact_face=4*cut, return_face_flux=True)
    faces = 4*np.arange(n)[:, None] + np.array([1, 2, 3])
    # The temperature vector contains both one-sided contact traces.
    target = nodes[faces + (faces > 4*cut)]
    if omega == 0:
        q = 1/(cut/n + resistance + (1-cut/n)/4)
        x = faces/(4*n)
        np.testing.assert_allclose(target, np.where(x < cut/n, 1-q*x, q*(1-x)/4), atol=1e-14)
        np.testing.assert_allclose(flux, q, rtol=1e-14)
    row = dict(cells=n, omega_rad_s=omega,
               temperature_error_K=float(abs(temperature-target).max()),
               heat_flux_error_W_m2=float(abs(power-flux[faces]).max()))
    if 0 < omega <= 1e-6:
        exact_t, exact_q = dc_slopes(faces/(4*n), cut/n, resistance,
                                     (k[0], k[-1]), (capacity[0], capacity[-1]))
        row["temperature_slope_error_K_s"] = float(abs(temperature.imag/omega-exact_t).max())
        row["heat_flux_slope_error_W_s_m2"] = float(abs(power.imag/omega-exact_q).max())
    return row


def main():
    rows = []
    for omega in (0., 1e-12, 1e-10, 1e-6, 1., 16.):
        sweep = [measure(n, omega) for n in (16, 32, 64)]
        for key in ('temperature_error_K', 'heat_flux_error_W_m2'):
            errors = np.array([row[key] for row in sweep])
            if omega < 1:
                np.testing.assert_array_less(errors, 1e-10)
            else:
                np.testing.assert_array_less(3*errors[1:], errors[:-1])
        if 0 < omega <= 1e-6:
            for key in ("temperature_slope_error_K_s", "heat_flux_slope_error_W_s_m2"):
                np.testing.assert_array_less([row[key] for row in sweep], 1e-11)
        rows.extend(sweep)
    print(json.dumps(dict(sources=[provenance(path) for path in
                                  (__file__, slab.__file__, rod.__file__)], rows=rows), indent=2))


if __name__ == "__main__":
    main()
