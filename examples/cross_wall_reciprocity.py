"""Check opposing wall-to-wall transfer phasors on a fixed two-material slab."""
import json

import numpy as np

import cross_wall_driven_harmonics as slab
from conduction_provenance import provenance


def cold_phasor(row, prefix=""):
    return np.array(row[prefix+"cold_gain_W_m2_K"])*np.exp(1j*np.array(row[prefix+"cold_phase_rad"]))


def wall_transfer(position, resistance=.3, *, with_sensitivity=False):
    """Cross-wall flux per unit drive from the two layer transfer matrices."""
    omega = 2*np.pi*np.array([1., 2.])[:, None]
    k = np.array([1., 4.])
    wave = np.sqrt(1j*omega*np.array([1., 2.])/k)
    z = wave*np.array([position, 1-position])
    c = np.cosh(z)
    impedance = np.sinh(z)/(wave*k)
    contact_weight = c[:, 0]*c[:, 1]
    gain = 1/(impedance[:, 0]*c[:, 1] + impedance[:, 1]*c[:, 0] + resistance*contact_weight)
    if with_sensitivity:
        # R*d(log G)/dR for the complex transfer gain G.
        return gain, -resistance*contact_weight*gain
    return gain


def study(solver, position=.5):
    rows = []
    previous = None
    previous_defects = None
    analytic = wall_transfer(position)
    for n in (16, 32, 64, 128):
        forward = slab.measure(n, solver, interface_position=position)
        reverse = slab.measure(n, solver, reverse=True, interface_position=position)
        direct, swapped = cold_phasor(forward), cold_phasor(reverse)
        exact = cold_phasor(forward, "exact_")
        exact_swapped = cold_phasor(reverse, "exact_")
        defects = abs((swapped-direct)/exact)
        reverse_errors = abs(swapped/exact_swapped-1)
        orders = None if previous is None else np.log2(previous/reverse_errors)
        exact_defect = float(max(abs(exact_swapped/exact-1)))
        formula_error = float(np.max(abs(np.array([exact, exact_swapped])/analytic-1)))
        defect_orders = None if previous_defects is None else np.log2(previous_defects/defects)
        passed = formula_error < 1e-12 and exact_defect < 1e-12 and reverse["max_modal_layer_balance_W_m2_K"] < 1e-9
        if n == 128:
            passed &= min(defect_orders) > 1.8
            passed &= max(defects) < .001 and max(reverse_errors) < .001 and min(orders) > 1.8
        rows.append(dict(interface_from_west_m=position, transfer_formula_relative_error=formula_error,
                         reciprocity_orders=None if defect_orders is None else defect_orders.tolist(),
                         columns=n, steps=4*n, solver=solver, relative_reciprocity_errors=defects.tolist(),
                         reverse_relative_complex_errors=reverse_errors.tolist(), exact_relative_defect=exact_defect,
                         reverse_gain_W_m2_K=reverse["cold_gain_W_m2_K"], reverse_phase_rad=reverse["cold_phase_rad"],
                         orders=orders.tolist() if orders is not None else None, passed=bool(passed)))
        previous = reverse_errors
        previous_defects = defects
    return rows


if __name__ == "__main__":
    runs = [row for position in (.25, .5, .75) for solver in ("fvm", "sph") for row in study(solver, position)]
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, frequencies_Hz=[1, 2], sources=[provenance(p) for p in
        (__file__, slab.__file__, slab.continuum.__file__, slab.periodic.__file__, slab.periodic.rod.__file__,
         slab.periodic.sph.__file__)], reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
         "PYTHONPATH=../TensorFVM/src:../TensorSPH/src python examples/cross_wall_reciprocity.py"),
         indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
