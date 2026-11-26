"""Partition affine internal heating between two isothermal walls across a resistive contact."""
import json

import numpy as np
from numpy.polynomial import Polynomial
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, sign, resistance):
    cut = 3*n//8
    lengths = (cut/n, 1-cut/n)
    conductivities = (1.3, 3.7)
    means = (1.1, .6)
    slopes = (sign*4., -sign*2.)
    # Start at zero west flux. The east temperature residual determines
    # the uniform flux correction through the full series resistance.
    q0, t0 = 0., 0.
    sources, fluxes, temperatures = [], [], []
    for i, (width, k, mean, slope) in enumerate(zip(lengths, conductivities, means, slopes)):
        source = Polynomial([mean-slope*width/2, slope])
        flux = source.integ(k=q0)
        temperature = (-flux/k).integ(k=t0)
        sources.append(source)
        fluxes.append(flux)
        temperatures.append(temperature)
        q0 = flux(width)
        t0 = temperature(width)
        if i == 0:
            t0 -= resistance*q0
    correction = t0/(sum(h/k for h, k in zip(lengths, conductivities))+resistance)
    local_faces = [np.arange(m+1)/n for m in (cut, n-cut)]
    centers = [(x[1:]+x[:-1])/2 for x in local_faces]
    path = [centers[0]/conductivities[0],
            lengths[0]/conductivities[0]+resistance+centers[1]/conductivities[1]]
    exact_t = np.concatenate([t(x)-correction*r for t, x, r in zip(temperatures, centers, path)])
    exact_q = correction+np.r_[fluxes[0](local_faces[0][:-1]), fluxes[1](local_faces[1])]
    np.testing.assert_allclose(exact_q[-1]-exact_q[0], np.dot(means, lengths), atol=1e-14, rtol=0.)
    source = np.concatenate([s(x) for s, x in zip(sources, centers)])
    contacts = np.zeros(n-1)
    contacts[cut-1] = resistance
    field, flux = sph.solve_harmonic_layered_slab(
        source, 1/n, (0., 0.), 0., conductivity=np.repeat(conductivities, [cut, n-cut]),
        volumetric_heat_capacity=1., contact_resistance_x=contacts, return_x_flux=True, backend="banded")
    q = flux.numpy()
    return dict(columns=n, gradient_sign=sign, resistance_m2_K_W=resistance,
                west_outward_W_m2=float(-q[0].real), east_outward_W_m2=float(q[-1].real),
                temperature_error_K=float(abs(field.numpy()-exact_t).max()),
                wall_flux_error_W_m2=float(abs((q-exact_q)[[0,-1]]).max()),
                all_face_flux_error_W_m2=float(abs(q-exact_q).max()),
                max_cell_balance_W_m2=float(abs(np.diff(q)-source/n).max()))


if __name__ == "__main__":
    runs = []
    for sign in (-1., 1.):
        for resistance in (0., .25):
            previous = None
            for n in (16, 32, 64, 128):
                row = measure(n, sign, resistance)
                errors = np.array([row["temperature_error_K"], row["wall_flux_error_W_m2"]])
                orders = np.log2(previous/errors) if previous is not None else None
                row["temperature_and_wall_flux_orders"] = orders.tolist() if orders is not None else None
                assert row["max_cell_balance_W_m2"] < 1e-10
                if n == 128:
                    np.testing.assert_array_less(1.7, orders)
                    assert row["all_face_flux_error_W_m2"] < 1e-5
                    assert row["temperature_error_K"] < 2e-5
                previous = errors
                runs.append(row)
    print(json.dumps(dict(reference="Piecewise cubic Fourier temperatures with two zero-temperature walls",
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_affine_contact_wall_heat.py",
                          sources=[provenance(p) for p in (__file__, sph.__file__)], runs=runs),
                     indent=2, allow_nan=False))
