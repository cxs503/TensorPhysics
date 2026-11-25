"""Compare five archived solver histories on identical spacetime cuts."""
import hashlib
import json
from pathlib import Path

import numpy as np

import sph_resistive_contact_step as model
from conduction_provenance import provenance

ARCHIVE = Path(__file__).resolve().parents[1] / "artifacts" / "verified_numerics_20261125"
TIMES = np.r_[0., .00025*2.**np.arange(12)]
RESISTANCES = (.05, .3)
GRIDS = (40, 80, 160, 320)
# Archive, solver labels, grid key, stored cut integrals, stored error triplet.
INPUTS = (
    ("sph_fvm", ("SPH", "FVM"), "cells", "interval_heat_J_m2", "max_west_interface_east_interval_errors_J_m2"),
    ("fem", ("FEM",), "elements", "interval_west_interface_east_heat_J_m2", "west_interface_east_heat_errors_J_m2"),
    ("dg", ("DG",), "columns", "interval_west_interface_east_heat_J_m2", "west_interface_east_heat_errors_J_m2"),
    ("dem", ("DEM",), "cells", "interval_west_interface_east_heat_J_m2", "west_interface_east_heat_error_J_m2"),
)


def compare(directory):
    reference = {r: model.interval_reference(TIMES, r, "dehoog") for r in RESISTANCES}
    histories = {}
    inputs = []
    for prefix, labels, grid, cut, summary in INPUTS:
        path = directory / f"{prefix}_resistive_contact_step.json"
        data = path.read_bytes()
        archive = json.loads(data)
        np.testing.assert_array_equal(archive["times_s"], TIMES)
        if not archive["passed"]:
            raise ValueError(f"{path.name} did not pass its original benchmark")
        inputs.append(dict(file=path.name, sha256=hashlib.sha256(data).hexdigest(),
                           sources=archive["sources"]))
        for study in archive["studies"]:
            resistance = study.get("contact_resistance_m2_K_W", study.get("resistance_m2_K_W"))
            archived_ref = study.get("reference_interval_heat_J_m2", study.get("reference_west_interface_east_heat_J_m2"))
            np.testing.assert_allclose(archived_ref, reference[resistance], rtol=0., atol=1e-13)
            for row in study["runs"]:
                name = row.get("solver", labels[0])
                n = row[grid]
                key = name, resistance, n
                if name not in labels or n not in GRIDS or key in histories or not row["passed"]:
                    raise ValueError(f"Invalid archived run {key} in {path.name}")
                heat = np.asarray(row[cut])
                if heat.shape != (len(TIMES)-1, 3) or not np.isfinite(heat).all():
                    raise ValueError(f"Invalid interval heat for {key}")
                error = abs(heat-reference[resistance]).max(axis=0)
                np.testing.assert_allclose(error, row[summary], rtol=0., atol=1e-13)
                histories[key] = heat

    groups = []
    solvers = sorted(name for _, labels, *_ in INPUTS for name in labels)
    for resistance in RESISTANCES:
        previous = {}
        for n in GRIDS:
            readings = []
            rows = []
            for name in solvers:
                heat = histories[name, resistance, n]
                error = abs(heat-reference[resistance]).max(axis=0)
                order = np.log2(previous[name]/error).tolist() if name in previous else []
                rows.append(dict(solver=name, max_cut_errors_J_m2=error.tolist(), orders=order,
                                 passed=bool((not order or min(order) > 0.) and
                                             (n != 320 or (min(order) > 1.85 and max(error) < 5e-5)))))
                readings.append(heat)
                previous[name] = error
            spread = np.ptp(readings, axis=0)
            worst = int(np.argmax(spread[:, 1]))
            groups.append(dict(contact_resistance_m2_K_W=resistance, cells=n, solvers=rows,
                               max_cut_spread_J_m2=spread.max(axis=0).tolist(),
                               worst_interface_interval_s=TIMES[worst:worst+2].tolist()))
    return dict(passed=all(row["passed"] for group in groups for row in group["solvers"]),
                cut_order=["west", "interface", "east"], times_s=TIMES.tolist(), groups=groups, inputs=inputs,
                reproduction="python examples/compare_resistive_contact_step.py",
                sources=[provenance(__file__), provenance(model.__file__)])


if __name__ == "__main__":
    result = compare(ARCHIVE)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
