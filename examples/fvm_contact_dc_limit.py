"""Verify reconstructed harmonic rod fields from DC through finite frequency."""
import json

import numpy as np
from tensorfvm.cht import rod

import periodic_contact_reference as slab
from conduction_provenance import provenance


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
    return dict(cells=n, omega_rad_s=omega,
                temperature_error_K=float(abs(temperature-target).max()),
                heat_flux_error_W_m2=float(abs(power-flux[faces]).max()))


def main():
    rows = []
    for omega in (0., 1e-10, 1., 16.):
        sweep = [measure(n, omega) for n in (16, 32, 64)]
        for key in ('temperature_error_K', 'heat_flux_error_W_m2'):
            errors = np.array([row[key] for row in sweep])
            if omega < 1:
                np.testing.assert_array_less(errors, 1e-10)
            else:
                np.testing.assert_array_less(3*errors[1:], errors[:-1])
        rows.extend(sweep)
    print(json.dumps(dict(sources=[provenance(path) for path in
                                  (__file__, slab.__file__, rod.__file__)], rows=rows), indent=2))


if __name__ == "__main__":
    main()
