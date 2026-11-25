"""Verify two-sided convection on a unit square of unit depth using a bilinear harmonic field."""
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n):
    h = 1 / n
    x = np.arange(n + 1) / n
    y = (np.arange(n) + 0.5) / n
    conductivity = 1.7
    # T = 300 + 3x - (2+x)y solves Laplace's equation and has varying normal flux.
    slope = -(2 + x)
    exact = 300 + 3*x[:, None] + slope[:, None]*y
    faces = []
    for position, outward in ((0, -1), (1, 1)):
        surface = 300 + 3*x + slope*position
        flux = -outward * conductivity * slope
        film = 3 + position + 2*x
        ambient = surface - flux / film
        # Series half-spacing conduction and convection over area h.
        conductance = h / (h / (2*conductivity) + 1/film)
        faces.append((conductance, ambient, film, surface, flux))
    field = contact_chain.lattice_network_temperatures(
        n + 1, n, 1/conductivity, exact[0], exact[-1],
        lower_exchange=faces[0][:2], upper_exchange=faces[1][:2],
    )
    measurements = []
    for side, (g, ambient, film, surface, flux) in zip((0, -1), faces):
        measured = g * (field[:, side] - ambient) / h
        # Trapezoidal integration along x, with half weight on each pinned end.
        power = h * (measured[0]/2 + measured[1:-1].sum() + measured[-1]/2)
        reference_power = float((flux[0] + flux[-1])/2)
        measurements.append(dict(reference_outward_power_W=reference_power, outward_power_W=float(power),
                                 power_error_W=float(abs(power - reference_power)),
                                 flux_error_W_m2=float(np.max(abs(measured - flux))),
                                 surface_temperature_error_K=float(np.max(abs(ambient + measured/film - surface)))))
    error = float(np.max(abs(field - exact)))
    passed = error < 1e-11 and all(f["surface_temperature_error_K"] < 1e-11
                                   and f["power_error_W"] < 1e-11 and f["flux_error_W_m2"] < 1e-11
                                   for f in measurements)
    return dict(intervals=n, field_error_K=error, lower=measurements[0], upper=measurements[1], passed=passed)


if __name__ == "__main__":
    runs = [measure(n) for n in (4, 15, 32, 63)]
    passed = all(run["passed"] for run in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_surface_exchange.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
