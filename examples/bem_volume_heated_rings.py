"""Manufactured polynomial Poisson modes in three conducting annuli and two storing contacts."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import harmonic_rings, layered_annulus


def measure(mode, formulation):
    edges = np.array([.4, .55, .9, 1.1])
    radius = edges[1:-1]
    k = np.array([1., 2.5, .8])
    resistance = np.array([.5, .9])
    fractions = np.array([.2, .7])
    capacity = np.array([.25, .8])
    coeff = np.array([.4+.1j, -.3+.6j, .8-.2j])
    density = -4*(mode+1)*k*coeff*edges[-1]**mode
    # T_i = A_i*r^m + B_i*r^(m+2). Choose the next A_i from the
    # resistive contact relation, including its dependence on the new radial flux.
    a = np.empty(3, dtype=complex)
    a[0] = .7-.3j

    def field(i, r):
        return a[i]*r**mode + coeff[i]*r**(mode+2)

    def flux(i, r):
        return -k[i]*(mode*a[i]*r**(mode-1) + (mode+2)*coeff[i]*r**(mode+1))

    sheets = []
    for i, r in enumerate(radius):
        incoming = flux(i, r)
        sheet = field(i, r)-fractions[i]*resistance[i]*incoming
        outer = (1-fractions[i])*resistance[i]
        a[i+1] = ((sheet-coeff[i+1]*(r**(mode+2)-outer*k[i+1]*(mode+2)*r**(mode+1)))
                  / (r**mode-outer*k[i+1]*mode*r**(mode-1)))
        sheets.append((sheet, incoming, flux(i+1, r)))
    t, qi, qo = np.asarray(sheets).T
    walls = [field(0, edges[0]), field(2, edges[-1])]
    target_rate = 2*np.pi*edges[[0,-1]]*[flux(0, edges[0]), flux(2, edges[-1])]
    target = np.r_[t, qi, qo, target_rate]
    rows = []
    previous = {}
    for n in (16, 32, 64, 128):
        response = harmonic_rings.assemble_ring_response(edges, k, resistance, n, mode=mode,
            inner_resistance_fractions=fractions, formulation=formulation, include_volume_sources=True)
        for omega in (1., 8., 24.):
            source = qo-qi + 1j*omega*capacity*t
            radii = [lo + (hi-lo)*np.array([.1, .5, .9]) for lo, hi in zip(edges, edges[1:])]
            samples = np.concatenate(radii) if formulation == "circular" else None
            result = response.solve(omega, walls, capacity, sheet_generation=source, volume_generation=density, sample_radii=samples)
            observed = np.r_[result.sheet_temperature, result.incoming_flux, result.outgoing_flux, result.wall_heat_rates]
            error = float(max(abs(observed/target-1)))
            order = np.log2(previous[omega]/error) if omega in previous and formulation == "polygonal" else None
            balance = float(max(abs(source+result.incoming_flux-result.outgoing_flux-1j*omega*capacity*result.sheet_temperature)))
            passed = (response.mode_leakage < 1e-9 and balance < 1e-10 and result.join_power_residual < 1e-10
                      and (error < 1e-9 if formulation == "circular" else n < 128 or (error < .01 and order > 1.7)))
            rows.append(dict(formulation=formulation, mode=mode, panels=n, omega=omega, max_relative_error=error,
                             order=order, sheet_balance_W_m2=balance, join_balance_W_m=result.join_power_residual, passed=bool(passed)))
            if samples is not None:
                exact_t = np.concatenate([field(i, r) for i, r in enumerate(radii)])
                exact_q = np.concatenate([flux(i, r) for i, r in enumerate(radii)])
                t_error = float(max(abs(result.sample_temperature-exact_t)))
                q_error = float(max(abs(result.sample_radial_flux-exact_q)))
                rows[-1].update(interior_temperature_max_error_K=t_error,
                                interior_radial_flux_max_error_W_m2=q_error,
                                passed=bool(passed and max(t_error, q_error) < 1e-9))
            previous[omega] = error
    return rows


if __name__ == "__main__":
    runs = [row for mode in range(3) for formulation in ("circular", "polygonal") for row in measure(mode, formulation)]
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(p) for p in
        (__file__, harmonic_rings.__file__, layered_annulus.__file__)],
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_volume_heated_rings.py"), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
