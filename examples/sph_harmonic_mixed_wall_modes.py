"""Manufactured 2-D harmonic slab with a prescribed temperature and a flux wall."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorsph import heat_conduction_2d as sph


K, CAPACITY, LOSS = 1.7, 2.2, .4
# Ascending powers of x. Distinct complex amplitudes drive both walls.
MODES = {
    0: [.3+.1j, .4-.2j, -.15+.1j, .2j],
    1: [.2j, .32+.1j, -.26j, .12],
    -2: [.17, -.1+.13j, .14, -.09j],
}


def reference(x, y, omega, *, x_width=0., y_width=0.):
    """Evaluate point values or exact averages over centered rectangular samples."""
    temperature, right_flux, source = np.zeros((3, len(y), len(x)), dtype=complex)
    for mode, coeff in MODES.items():
        wave = 2*np.pi*mode
        phase = np.exp(1j*wave*y[:, None])*np.sinc(mode*y_width)
        poly = np.polynomial.Polynomial(coeff)
        # For a cubic the centered average terminates after the second derivative.
        poly = poly + poly.deriv(2)*(x_width**2/24)
        temperature += phase*poly(x)
        right_flux -= K*phase*poly.deriv()(x)
        source += phase*((LOSS+1j*omega*CAPACITY+K*wave**2)*poly(x)-K*poly.deriv(2)(x))
    return temperature, right_flux, source


def measure(n, omega, fixed_left):
    dx = 1/n
    centers = (np.arange(n)+.5)*dx
    cuts = np.arange(n+1)*dx
    exact, _, source = reference(centers, centers, omega)
    face_t, face_q, _ = reference(cuts, centers, omega)
    walls = (face_t[:, 0], None) if fixed_left else (None, face_t[:, -1])
    inward = (0., -face_q[:, -1]) if fixed_left else (face_q[:, 0], 0.)
    t, q = sph.solve_harmonic_slab_2d(source, dx, walls, omega, conductivity=K,
        volumetric_heat_capacity=CAPACITY, heat_loss_coefficient=LOSS,
        x_wall_inward_flux=inward, return_x_flux=True)
    side = "left" if fixed_left else "right"
    index, sign = (0, 1) if fixed_left else (-1, -1)
    wall_flux = sph.recover_dirichlet_wall_flux_2d(
        t, dx, K, sign*q[index], x_wall_temperatures=walls, side=side).numpy()
    exact_wall_flux = sign*face_q[:, index]
    t, q = t.numpy(), q.numpy()
    balance = dx*((LOSS+1j*omega*CAPACITY)*t-source).mean(axis=0)+np.diff(q)
    return dict(columns=n, omega_rad_s=omega, fixed_wall="left" if fixed_left else "right",
                fixed_wall_row_flux_error_W_m2=float(abs(wall_flux-exact_wall_flux).max()),
                temperature_error_K=float(abs(t-exact).max()),
                section_flux_error_W_m2=float(abs(q-face_q.mean(axis=0)).max()),
                prescribed_mean_flux_error_W_m2=float(abs(q-face_q.mean(axis=0))[-1 if fixed_left else 0]),
                column_balance_residual_W_m2=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    for omega in (1e-6, 3.):
        for fixed_left in (True, False):
            previous = None
            for n in (8, 16, 32, 64):
                row = measure(n, omega, fixed_left)
                errors = np.array([row["temperature_error_K"], row["section_flux_error_W_m2"],
                                   row["fixed_wall_row_flux_error_W_m2"]])
                orders = np.log2(previous/errors).tolist() if previous is not None else []
                row.update(temperature_and_flux_orders=orders[:2],
                           fixed_wall_row_flux_order=orders[2] if orders else None,
                           passed=bool(row["column_balance_residual_W_m2"] < 1e-10 and
                                       row["prescribed_mean_flux_error_W_m2"] < 1e-10 and
                                       (not orders or min(orders) > 0) and
                                       (n != 64 or min(orders) > 1.7)))
                previous = errors
                rows.append(row)
    print(json.dumps(dict(reference="Cubic polynomials in x times periodic modes 0, 1 and -2; unit square",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_harmonic_mixed_wall_modes.py",
        runs=rows, sources=[provenance(__file__), provenance(sph.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
