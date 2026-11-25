"""Steady and harmonic heat partition from a strip spanning two materials and a resistive contact."""
import argparse
import json
import math

import numpy as np
from scipy.linalg import expm
from tensorfvm.cht import rod

from conduction_provenance import provenance


def contact_admittance_reference(conductivities, capacities, resistance, omega, trace_capacity, trace_power,
                                 strip_widths=(0., 0.), generation=0., strip_gaps=(0., 0.), transverse_wavenumber=0.):
    """Outward wall powers and trace jump for heated segments at given distances from the contact."""
    k = np.asarray(conductivities)
    z = np.sqrt(transverse_wavenumber**2 + 1j*omega*np.asarray(capacities)/k)
    attenuation = np.exp(-z/2)
    denominator = -np.expm1(-z)
    diagonal = k*z*(1+attenuation**2)/denominator + 1j*omega*np.asarray(trace_capacity)
    contact_matrix = np.array([[1., -1.], [-1., 1.]])/resistance
    # Integrate the Dirichlet wall and contact flux kernels over each heated
    # segment. Distances are measured from the interface, with half length 0.5.
    widths = np.asarray(strip_widths)
    near = np.asarray(strip_gaps)
    far = near + widths
    heated_fraction = -np.expm1(-z*widths)
    scaled_source = generation * heated_fraction / (z*denominator)
    contact_load = scaled_source * np.exp(-z*near) * (-np.expm1(-z*(1-near-far)))
    wall_load = scaled_source * (-np.expm1(-z*(near+far))) * np.exp(-z*(0.5-far))
    traces = np.linalg.solve(np.diag(diagonal) + contact_matrix, trace_power + contact_load)
    return np.r_[wall_load + 2*k*z*traces*attenuation/denominator, traces[0]-traces[1]]


def harmonic_reference(start, end, conductivities, capacities, generation, resistance, omega, trace_capacity=(0., 0.), trace_power=(0., 0.)):
    # Propagate (T, rightward power, 1) exactly within each constant-source
    # segment. The final component makes the volumetric source an affine load.
    transfer = np.eye(3)
    jump = np.array([[1., -resistance, 0.], [0., 1., 0.], [0., 0., 1.]])
    # Each trace withdraws i*omega*C*T before or after the resistive drop.
    left_store, right_store = (np.eye(3, dtype=complex) for _ in trace_capacity)
    left_store[1, 0], right_store[1, 0] = -1j*omega*np.asarray(trace_capacity)
    left_store[1, 2], right_store[1, 2] = trace_power
    edges = sorted({0., start, 0.5, end, 1.})
    for left, right in zip(edges[:-1], edges[1:]):
        layer = int(left >= 0.5)
        source = generation if left >= start and right <= end else 0.
        generator = [[0., -1/conductivities[layer], 0.],
                     [-1j*omega*capacities[layer], 0., source], [0., 0., 0.]]
        transfer = expm((right-left)*np.array(generator)) @ transfer
        if right == 0.5:
            at_contact = transfer.copy()
            transfer = right_store @ jump @ left_store @ transfer
    # Both ends have zero temperature phasor. Solve for the left-end power.
    left_power = -transfer[0, 2]/transfer[0, 1]
    initial = np.array([0., left_power, 1.])
    return np.array([-left_power, (transfer @ initial)[1],
                     resistance*(left_store @ at_contact @ initial)[1]])


def strip_segments(start, end):
    """Heated widths and gaps from the contact in the two half rods."""
    near = np.clip([0.5-end, start-0.5], 0., 0.5)
    far = np.clip([0.5-start, end-0.5], 0., 0.5)
    return far-near, near


def measure(cells, mirrored, omega=0., trace_capacity=(0., 0.), trace_power=(0., 0.), generation=12., detached=False):
    interface, resistance = 0.5, 0.35
    start, end = (3/8, 5/6) if mirrored else (1/6, 5/8)
    if detached:
        start, end = (5/8, 5/6) if mirrored else (1/6, 3/8)
    k_left, k_right = (4., 1.) if mirrored else (1., 4.)
    faces = np.linspace(0., 1., cells+1)
    widths = np.diff(faces)
    mid = cells // 2
    # Unit area. F(x) is total generated power to the left of x.
    def generated(x):
        return generation * np.clip(x-start, 0., end-start)

    def integral(x):
        return generation / 2 * (np.maximum(x-start, 0.)**2 - np.maximum(x-end, 0.)**2)

    contacts = np.zeros(cells-1)
    contacts[mid-1] = resistance
    source = np.diff(generated(faces)) / widths
    if omega:
        capacities = (2.5, 1.25) if mirrored else (1.25, 2.5)
        capacity = np.repeat(capacities, mid)
        sheet = np.zeros(cells-1)
        sheet[mid-1] = sum(trace_capacity)
        fraction = trace_capacity[0]/sum(trace_capacity) if sum(trace_capacity) else 0.5
        applied = np.zeros((2, cells-1), dtype=complex)
        applied[:, mid-1] = trace_power
        field, _, west, east, (upstream, downstream) = rod.solve_harmonic_stepped_rod(
            widths, np.ones(cells), np.repeat([k_left, k_right], mid), capacity,
            (0., 0.), omega, source=source, flux_order=2, contact_resistance=contacts, return_traces=True,
            interface_heat_capacity=sheet, interface_capacity_left_fraction=fraction, trace_power=applied,
        )
        actual = np.array([-west[0], east[-1], upstream[mid-1]-downstream[mid-1]])
        heated_widths, gaps = strip_segments(start, end)
        exact = contact_admittance_reference((k_left, k_right), capacities, resistance, omega,
                                             trace_capacity, trace_power, heated_widths, generation, gaps)
        trace_t = np.array([upstream[mid-1], downstream[mid-1]])
        crossing = actual[2]/resistance
        trace_net = np.array([east[mid-1]-crossing, crossing-west[mid]])
        return {
            "cells": cells, "strip_bounds_m": [start, end], "omega_rad_s": omega,
            "max_trace_balance_error_W": max(abs(trace_net + trace_power - 1j*omega*trace_t*trace_capacity)),
            "wall_outward_amplitude_W": abs(actual[:2]).tolist(),
            "reference_wall_amplitude_W": abs(exact[:2]).tolist(),
            "contact_jump_amplitude_K": abs(actual[2]), "reference_jump_amplitude_K": abs(exact[2]),
            "wall_and_jump_phase_rad": np.angle(actual).tolist(),
            "reference_wall_and_jump_phase_rad": np.angle(exact).tolist(),
            "max_wall_relative_error": max(abs(actual[:2]/exact[:2]-1)),
            "max_wall_power_error_W": max(abs(actual[:2]-exact[:2])),
            "contact_jump_error_K": abs(actual[2]-exact[2]),
            "max_phase_error_rad": max(abs(np.angle(actual/exact))),
            "max_cell_balance_error_W": max(abs(east-west+widths*(1j*omega*capacity*field-source))),
        }

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega", type=float, default=0., help="angular frequency in rad/s (0 for steady)")
    omega = parser.parse_args().omega
    if not math.isfinite(omega) or omega < 0:
        parser.error("omega must be finite and nonnegative")
    metrics = (("max_wall_power_error_W", "contact_jump_error_K") if omega else
               ("max_face_temperature_error_K", "max_face_power_error_W"))
    runs = []
    for mirrored in (False, True):
        previous = None
        for cells in (16, 32, 64, 128):
            row = measure(cells, mirrored, omega)
            passed = row["max_cell_balance_error_W"] < 1e-10
            for metric in metrics:
                if previous:
                    order = math.log2(previous[metric]/row[metric])
                    row[metric+"_order"] = order
                    passed &= order > 1.85
            if cells == 128:
                passed &= max(row[key] for key in metrics) < 2e-4
                if omega:
                    passed &= row["max_phase_error_rad"] < 5e-4
            row["passed"] = bool(passed)
            runs.append(row)
            previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps({
        "passed": passed, "runs": runs, "sources": [provenance(__file__), provenance(rod.__file__)],
        "reproduction": "PYTHONPATH=../TensorFVM/src python examples/fvm_strip_heater_across_contact.py"
                        + (f" --omega {omega}" if omega else ""),
    }, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
