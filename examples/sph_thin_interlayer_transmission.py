"""Verify modal section heat through a thin layer bounded by two resistive contacts."""
import json
import math

import numpy as np
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, beta, backend):
    dx = 1 / n
    # Fixed physical thickness of 1/32 m, with one column at the coarsest level.
    first, last = n // 2, n // 2 + n // 32
    k = np.ones(n)
    k[first:last] = 4.
    capacity = np.full(n, 1.25)
    capacity[first:last] = 2.5
    contacts = np.zeros(n-1)
    contacts[[first-1, last-1]] = [0.12, 0.23]
    bulk = dx/k
    resistance = np.cumsum(bulk) - bulk/2 + np.r_[0., np.cumsum(contacts)]
    total = sum(bulk) + sum(contacts)
    # For the exp(i beta y) amplitude, T(x) = T_left - q * R(x).
    # Thus the axial flux q is continuous and each trace jumps by q*R_contact.
    omega, west, q = 3., 0.4+0.3j, 0.7-0.2j
    target = west - q*resistance
    source = (1j*omega*capacity + k*beta**2) * target
    field, cuts = sph.solve_harmonic_layered_slab(
        source, dx, (west, west-q*total), omega, conductivity=k,
        volumetric_heat_capacity=capacity, contact_resistance_x=contacts,
        h_factor=1.6, transverse_wavenumber=beta, backend=backend,
        split_interface_links=True, return_x_flux=True,
    )
    errors = np.abs(cuts.numpy()/q-1)
    return {
        "columns": n, "interlayer_columns": last-first, "beta_rad_m": beta, "backend": backend,
        "temperature_error_K": float(abs(field.numpy()-target).max()),
        "contact_flux_relative_errors": errors[[first, last]].tolist(),
        "max_cut_flux_relative_error": float(errors.max()),
        "worst_cut_x_m": int(errors.argmax())*dx,
        "wall_flux_relative_errors": errors[[0, -1]].tolist(),
    }


def main():
    runs = []
    for beta in (0., 2*math.pi, 4*math.pi):
        previous = None
        grids = (32, 64, 128, 256, 512, 1024, 2048)
        if beta:
            grids += (4096, 8192)
        for n in grids:
            row = measure(n, beta, "banded")
            error = row["max_cut_flux_relative_error"]
            passed = np.isfinite(error) and np.isfinite(row["temperature_error_K"])
            if beta == 0:
                passed &= max(row["temperature_error_K"], error) < 1e-9
            elif previous is not None:
                row["flux_error_order"] = math.log2(previous["max_cut_flux_relative_error"]/error)
                row["temperature_error_order"] = math.log2(previous["temperature_error_K"]/row["temperature_error_K"])
                passed &= all(new < old for new, old in zip(row["contact_flux_relative_errors"],
                                                            previous["contact_flux_relative_errors"]))
                # Until the layer is wider than two stencil radii the regions
                # adjacent to its two contacts overlap. Track the worst cut
                # throughout, and grade its rate once the layer is resolved.
                if row["interlayer_columns"] >= 8:
                    passed &= row["flux_error_order"] > 0.8
                passed &= row["temperature_error_order"] > 1.8
            if n == 32:
                dense = measure(n, beta, "torch")
                np.testing.assert_allclose(dense["contact_flux_relative_errors"],
                                           row["contact_flux_relative_errors"], atol=1e-12, rtol=1e-10)
                row["dense_backend"] = dense
            if beta and n == grids[-1]:
                passed &= error < 1e-3 and row["temperature_error_K"] < 1e-4
            row["passed"] = bool(passed)
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(path) for path in (__file__, sph.__file__)],
        "reproduction": "OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_thin_interlayer_transmission.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
