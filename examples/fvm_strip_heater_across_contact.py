"""Steady heat partition from a strip spanning two materials and a resistive contact."""
import json
import math

import numpy as np
from tensorfvm.cht import rod

from conduction_provenance import provenance


def measure(cells, mirrored):
    interface, resistance, generation = 0.5, 0.35, 12.
    start, end = (3/8, 5/6) if mirrored else (1/6, 5/8)
    k_left, k_right = (4., 1.) if mirrored else (1., 4.)
    faces = np.linspace(0., 1., cells+1)
    widths = np.diff(faces)
    mid = cells // 2
    # Unit area. F(x) is total generated power to the left of x.
    def generated(x):
        return generation * np.clip(x-start, 0., end-start)

    def integral(x):
        return generation / 2 * (np.maximum(x-start, 0.)**2 - np.maximum(x-end, 0.)**2)

    # q(x) = q(0) + F(x). The two bulk temperature drops plus the
    # jump R*q(interface) sum to zero between equally hot ends.
    weighted = integral(interface)/k_left + (integral(1.)-integral(interface))/k_right
    q0 = -(weighted + resistance*generated(interface)) / (interface/k_left + resistance + (1-interface)/k_right)
    exact_flux = q0 + generated(faces)
    exact_jump = resistance * exact_flux[mid]
    drop_left = (q0 * faces[:mid+1] + integral(faces[:mid+1])) / k_left
    drop_right = (q0 * (faces[mid:]-interface) + integral(faces[mid:])-integral(interface)) / k_right
    wall = 300.
    left_faces = wall - drop_left
    right_faces = left_faces[-1] - exact_jump - drop_right

    contacts = np.zeros(cells-1)
    contacts[mid-1] = resistance
    source = np.diff(generated(faces)) / widths
    _, _, west_power, east_power, (upstream, downstream) = rod.solve_steady_stepped_rod(
        widths, np.ones(cells), np.repeat([k_left, k_right], mid), np.ones(cells),
        (wall, wall), source=source, flux_order=2, contact_resistance=contacts, return_traces=True,
    )
    # upstream and downstream give temperatures on opposite sides of every interior face.
    expected_upstream = np.r_[left_faces[1:], right_faces[1:-1]]
    expected_downstream = np.r_[left_faces[1:-1], right_faces[:-1]]
    jump = upstream[mid-1] - downstream[mid-1]
    return {
        "cells": cells, "strip_bounds_m": [start, end], "conductivity_W_m_K": [k_left, k_right],
        "wall_outward_W": [-west_power[0], east_power[-1]],
        "reference_wall_outward_W": [-q0, exact_flux[-1]],
        "contact_jump_K": jump, "reference_contact_jump_K": exact_jump,
        "max_face_temperature_error_K": max(abs(upstream-expected_upstream).max(),
                                            abs(downstream-expected_downstream).max()),
        "max_face_power_error_W": max(abs(west_power-exact_flux[:-1]).max(),
                                      abs(east_power-exact_flux[1:]).max()),
        "max_cell_balance_error_W": abs(east_power-west_power-source*widths).max(),
    }


def main():
    runs = []
    for mirrored in (False, True):
        previous = None
        for cells in (16, 32, 64, 128):
            row = measure(cells, mirrored)
            passed = row["max_cell_balance_error_W"] < 1e-10
            for metric in ("max_face_temperature_error_K", "max_face_power_error_W"):
                if previous:
                    order = math.log2(previous[metric]/row[metric])
                    row[metric+"_order"] = order
                    passed &= order > 1.85
            if cells == 128:
                passed &= max(row["max_face_temperature_error_K"], row["max_face_power_error_W"]) < 2e-4
            row["passed"] = bool(passed)
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(__file__), provenance(rod.__file__)],
        "reproduction": "PYTHONPATH=../TensorFVM/src python examples/fvm_strip_heater_across_contact.py",
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
