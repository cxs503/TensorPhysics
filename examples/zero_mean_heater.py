"""Heat transfer from a linear source phasor with zero integral across the layer."""
import argparse
import json
from inspect import getfile

import numpy as np

import finite_heater_layer as layer
from conduction_provenance import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-heater", action="store_true", help="start with two cells in the active layer")
    coarse = parser.parse_args().coarse_heater
    thickness, heater_k = .04, 2.
    runs = []
    for fraction in (.15, .85):
        exact, _ = layer.reference(thickness, fraction, heater_k, mean_source=0., source_slope=1.)
        previous = {}
        for n in (8, 16, 32, 64, 128):
            active = n//4 if coarse else n//2
            for name, solve in (("FEM", layer.fem_response), ("FVM", layer.numerical)):
                values, balance = solve(n, thickness, fraction, heater_k, mean_source=0., source_slope=1., heater_cells=active)
                errors = abs(values-exact)/abs(exact)
                orders = np.log2(previous[name]/errors) if name in previous else None
                passed = np.isfinite(errors).all() and balance < 1e-11
                if orders is not None:
                    passed &= min(orders) > 1.8
                runs.append(dict(solver=name, heater_cells=active, outer_cells_per_side=n, fraction=fraction,
                                 relative_errors_mean_T_left_q_right_q=errors.tolist(), balance_W_m2=balance,
                                 exact_outward_flux_magnitudes_W_m2=abs(exact[1:]).tolist(),
                                 orders=orders.tolist() if orders is not None else None, passed=bool(passed)))
                previous[name] = errors
    print(json.dumps(dict(thickness_m=thickness, heater_k_W_mK=heater_k, runs=runs,
                     sources=[provenance(f) for f in (__file__, layer.__file__, layer.thermal.__file__,
                              layer.energy.__file__, getfile(layer.sheet_reference))]), indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
