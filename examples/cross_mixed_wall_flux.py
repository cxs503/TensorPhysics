"""Compare fixed-wall flux convergence using the same manufactured mixed-boundary slab."""
import json

import numpy as np
from tensorfvm.cht import energy

import sph_harmonic_mixed_wall_modes as slab
from conduction_provenance import provenance


def fvm(n, omega, fixed_left):
    dx = 1/n
    widths = np.full(n, dx)
    centers = (np.arange(n)+.5)*dx
    t_exact, _, source = slab.reference(centers, centers, omega, x_width=dx, y_width=dx)
    t_wall, q_wall, _ = slab.reference(np.array([0., 1.]), centers, omega, y_width=dx)
    fixed, loaded = ("west", "east") if fixed_left else ("east", "west")
    index, sign = (0, 1) if fixed_left else (1, -1)
    prescribed = -sign*q_wall[:, 1-index]
    bc = {fixed: dict(type="dirichlet", values=t_wall[:, index]),
          loaded: dict(type="neumann", values=prescribed)}
    t, wall, inside = energy.solve_harmonic_conduction(
        widths, widths, slab.K, slab.CAPACITY, bc, omega, source=source,
        heat_loss_coefficient=slab.LOSS, dirichlet_flux_order=2, periodic_y=True,
        return_interior_power=True)
    px = np.column_stack((wall["west"], inside["x"], -wall["east"]))
    py = np.vstack((wall["south"], inside["y"], -wall["north"]))
    balance = ((1j*omega*slab.CAPACITY+slab.LOSS)*t-source)*dx**2
    balance += np.diff(px, axis=1)+np.diff(py, axis=0)
    np.testing.assert_allclose(wall[loaded]/dx, prescribed, atol=1e-12, rtol=0)
    np.testing.assert_array_less(abs(balance).max(), 1e-11)
    return dict(temperature_error_K=float(abs(t-t_exact).max()),
                fixed_wall_row_flux_error_W_m2=float(abs(wall[fixed]/dx-sign*q_wall[:, index]).max()),
                max_cell_balance_error_W=float(abs(balance).max()))


def main():
    runs = []
    for omega in (1e-6, 3.):
        for fixed_left in (True, False):
            previous = {}
            for n in (8, 16, 32, 64):
                for method, measure in (("FVM", fvm), ("SPH", slab.measure)):
                    row = measure(n, omega, fixed_left)
                    errors = np.array([row["temperature_error_K"], row["fixed_wall_row_flux_error_W_m2"]])
                    orders = np.log2(previous[method]/errors) if method in previous else None
                    if orders is not None:
                        np.testing.assert_array_less(1.7 if n == 64 else 0., orders)
                    row.update(solver=method, columns=n, omega_rad_s=omega,
                               fixed_wall="left" if fixed_left else "right",
                               temperature_and_wall_flux_orders=None if orders is None else orders.tolist())
                    previous[method] = errors
                    runs.append(row)
    print(json.dumps(dict(reference="FVM uses exact cell and face averages. SPH uses point samples.",
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_mixed_wall_flux.py",
                          sources=[provenance(p) for p in (__file__, slab.__file__, slab.sph.__file__, energy.__file__)],
                          runs=runs), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
