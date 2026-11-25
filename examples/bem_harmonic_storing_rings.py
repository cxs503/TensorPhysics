"""Periodic response of two storing rings with quasi-steady conduction in three annular materials."""
import json
import math

import numpy as np
from tensorbem import layered_annulus as bem

from conduction_provenance import provenance


RADII = (.4, .55, .75, .9, 1.1)  # bore, sheet, artificial join, sheet, exterior
K = (1., 2.5, .8)
CONTACT = (.5, .9)
CAPACITY = (.25, .8)
WALLS = np.array([1., .3j])


def ring_response(n, formulation):
    # Each pair of annuli maps its wall temperatures and sheet generation
    # into incoming/outgoing powers and the sheet temperature by linearity.
    maps = []
    for i in range(2):
        radii = RADII[2*i:2*i+3]
        columns = []
        for left, right, generation in np.eye(3):
            result = bem.solve_annulus_contact(radii, K[i:i+2], (left, right), CONTACT[i], n,
                                               formulation=formulation, interface_source=generation)
            columns.append([result.heat_rates[0], result.heat_rates[2], result.sheet_temperature.mean(),
                            result.interface_flux.mean(), result.outer_interface_flux.mean()])
        maps.append(np.asarray(columns).T)
    return maps


def solve_coupled(maps, omega):
    def outputs(unknown):
        join, source1, source2 = unknown
        a = maps[0] @ [WALLS[0], join, source1]
        b = maps[1] @ [join, WALLS[1], source2]
        residual = np.array([a[1]-b[0], source1+1j*omega*CAPACITY[0]*a[2],
                             source2+1j*omega*CAPACITY[1]*b[2]])
        return residual, np.stack((a[2:], b[2:]))

    base, _ = outputs(np.zeros(3))
    matrix = np.column_stack([outputs(unit)[0]-base for unit in np.eye(3)])
    unknown = np.linalg.solve(matrix, -base)
    residual, fields = outputs(unknown)
    return fields, float(abs(residual[0]))


def continuum(omega):
    # Exact radial resistances in K m/W, with heat rates per unit axial length.
    a, r1, _, r2, b = RADII
    surface = 2*np.pi*np.array([r1, r2])
    half_contact = np.asarray(CONTACT)/(2*surface)
    resistance = np.log([r1/a, r2/r1, b/r2])/(2*np.pi*np.array(K))
    resistance += [half_contact[0], half_contact.sum(), half_contact[1]]
    g0, g1, g2 = 1/resistance
    matrix = np.array([[g0+g1, -g1], [-g1, g1+g2]]) + 1j*omega*np.diag(surface*CAPACITY)
    temp = np.linalg.solve(matrix, WALLS*[g0, g2])
    power = -np.diff(np.r_[WALLS[0], temp, WALLS[1]])/resistance
    return np.column_stack((temp, power[:-1]/surface, power[1:]/surface))


if __name__ == "__main__":
    runs = []
    for formulation in ("circular", "polygonal"):
        previous = {}
        for n in (16, 32, 64):
            maps = ring_response(n, formulation)
            for omega in (1., 8., 24.):
                fields, imbalance = solve_coupled(maps, omega)
                expected = continuum(omega)
                ratio = fields/expected
                error = float(abs(ratio-1).max())
                order = None if omega not in previous or formulation == "circular" else math.log2(previous[omega]/error)
                sheet_balance = float(abs(fields[:, 1]-fields[:, 2]-1j*omega*np.asarray(CAPACITY)*fields[:, 0]).max())
                passed = imbalance < 1e-10 and sheet_balance < 1e-10 and (error < 1e-9 if formulation == "circular"
                                                else n < 64 or (order > 1.7 and error < .01))
                runs.append(dict(formulation=formulation, panels=n, omega=omega, max_relative_phasor_error=error,
                                 order=order, amplitude_ratios=abs(ratio).tolist(), phase_errors_deg=np.angle(ratio,deg=True).tolist(),
                                 reference_amplitudes=abs(expected).tolist(), reference_phases_deg=np.angle(expected,deg=True).tolist(),
                                 join_power_residual_W_m=imbalance, sheet_balance_residual_W_m2=sheet_balance, passed=passed))
                previous[omega] = error
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, model="axisymmetric, zero bulk capacity, positive areal storage at each ring",
                          columns=["sheet temperature", "incoming radial flux", "outgoing radial flux"], runs=runs,
                          sources=[provenance(p) for p in (__file__, bem.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_harmonic_storing_rings.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
