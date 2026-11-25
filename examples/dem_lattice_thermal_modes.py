"""Steady anisotropic particle lattice driven by row-dependent end temperatures."""
import argparse
import json

import numpy as np
from tensordem.thermal import contact_chain

from conduction_provenance import provenance


def measure(n, mode, ky, periodic=False):
    # Unit square and unit depth, equally spaced particles with insulating
    # transverse ends half a spacing outside the outer rows.
    dx = 1/n
    x = np.arange(n+1)[:, None]*dx
    y = (np.arange(n)+.5)*dx
    kx = 1.5
    wave = (2 if periodic else 1)*np.pi*mode
    phase = .73 if periodic else 0.
    lateral = np.cos(wave*y+phase)
    west, east = 1.+.6*lateral, .2-.3*lateral
    field = contact_chain.lattice_network_temperatures(n+1, n, (1/kx, 1/ky), west, east, periodic_y=periodic)
    continuum = wave*np.sqrt(ky/kx)
    discrete = 2*np.arcsinh(np.sqrt(ky/kx)*np.sin(wave*dx/2))/dx

    def profile(rate, position):
        return (.6*np.sinh(rate*(1-position))-.3*np.sinh(rate*position))/np.sinh(rate)

    def temperature(rate):
        return 1.-.8*x+profile(rate, x)*lateral

    # Compare link flux densities at link midpoints to -K grad(T).
    mid = (x[:-1]+x[1:])/2
    gradient = -continuum*(.6*np.cosh(continuum*(1-mid))+.3*np.cosh(continuum*mid))/np.sinh(continuum)
    qx = -kx*np.diff(field, axis=0)/dx
    qy = -ky*np.diff(field, axis=1)/dx
    exact_x = kx*(.8-gradient*lateral)
    exact_y = ky*wave*profile(continuum, x)*np.sin(wave*(y[:-1]+dx/2))
    seam = {}
    if periodic:
        qy = -ky*(np.roll(field, -1, axis=1)-field)/dx
        exact_y = ky*wave*profile(continuum, x)*np.sin(wave*(y+dx/2)+phase)
        divergence = qy[1:-1]-np.roll(qy[1:-1], 1, axis=1)
        seam["max_seam_flux_error_W_m2"] = float(abs(qy[:, -1]-exact_y[:, -1]).max())
    else:
        divergence = np.diff(np.pad(qy[1:-1], ((0, 0), (1, 1))), axis=1)
    net = np.diff(qx, axis=0)+divergence
    return dict(**seam, intervals=n, transverse_mode=mode, kx=kx, ky=ky,
                max_discrete_temperature_error_K=float(abs(field-temperature(discrete)).max()),
                max_continuum_temperature_error_K=float(abs(field-temperature(continuum)).max()),
                max_x_flux_error_W_m2=float(abs(qx-exact_x).max()),
                max_y_flux_error_W_m2=float(abs(qy-exact_y).max()),
                max_free_particle_balance_W=float(dx*abs(net).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periodic", action="store_true", help="drive a phase-shifted Fourier mode across the periodic seam")
    periodic = parser.parse_args().periodic
    runs = []
    grids = (8, 16, 32, 64, 128, 256, 512) if periodic else (8, 16, 32, 64, 128, 256)
    for mode in (1, 2):
        for ky in (.6, 2.4):
            previous = None
            for n in grids:
                row = measure(n, mode, ky, periodic)
                errors = np.array([row[key] for key in ("max_continuum_temperature_error_K",
                                                       "max_x_flux_error_W_m2", "max_y_flux_error_W_m2")])
                if periodic:
                    errors = np.r_[errors, row["max_seam_flux_error_W_m2"]]
                orders = [] if previous is None else np.log2(previous/errors).tolist()
                row.update(orders=orders, passed=bool(row["max_discrete_temperature_error_K"] < 1e-11
                           and row["max_free_particle_balance_W"] < 1e-11 and (not orders or min(orders) > 0.)
                           and (n < grids[-1] or min(orders) > 1.85)))
                runs.append(row)
                previous = errors
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(contact_chain.__file__)],
                          reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_lattice_thermal_modes.py" + (" --periodic" if periodic else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
