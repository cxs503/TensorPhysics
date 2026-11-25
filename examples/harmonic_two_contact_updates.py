"""Verify coupled finite resistance changes in a three-material harmonic rod."""
import argparse
from dataclasses import replace
import json

import numpy as np
from scipy.linalg import expm

from conduction_provenance import provenance


LENGTH = np.array([.28, .39, .33])
AREA = np.array([.8, 1.3, .95])
K = np.array([.9, 3.7, 1.8])
C = np.array([1.6, .85, 2.1])
WALL_C = np.array([.28, .63])
BASE = np.array([.32, .18])


def response(n, omega, resistance, probe_contacts=False):
    from tensorfvm.cht import rod

    indices = np.array([n-1, 2*n-1])
    contacts = np.zeros(3*n-1)
    contacts[indices] = resistance*np.minimum(AREA[:-1], AREA[1:])
    ends, drops = [], []
    drive_count = 4 if probe_contacts else 2
    for drive in np.eye(4)[:drive_count]:
        # A contact probe adds +1 W and -1 W on its two material traces.
        left = np.zeros_like(contacts)
        left[indices] = drive[2:]
        _, t, _, _, (west, east) = rod.solve_harmonic_stepped_rod(
            np.repeat(LENGTH/n, n), np.repeat(AREA, n), np.repeat(K, n), np.repeat(C, n),
            drive[:2], omega, end_types=("power", "power"), end_areal_heat_capacity=WALL_C,
            flux_order=2, contact_resistance=contacts, trace_power=(left, -left), return_traces=True)
        ends.append(t[[0, -1]])
        drops.append((west-east)[indices])
    return np.column_stack(ends), np.column_stack(drops)


def fem_response(n, omega, resistance, probe_contacts=False):
    import torch
    from tensorfem import thermal

    # Give each layer its own nodes, including two distinct temperatures per contact.
    edges = np.r_[0., np.cumsum(LENGTH)]
    x = np.concatenate([np.linspace(a, b, n+1) for a, b in zip(edges[:-1], edges[1:])])
    starts = (np.arange(3)[:, None]*(n+1)+np.arange(n)).ravel()
    elements = torch.from_numpy(np.column_stack((starts, starts+1)))
    cuts = np.arange(1, 3)*(n+1)-1
    last = len(x)-1
    model = thermal.ThermalModel(
        nodes=torch.from_numpy(x[:, None]), elements=elements, conductivity=torch.from_numpy(np.repeat(K, n)),
        specific_heat=torch.from_numpy(np.repeat(C, n)), density=1., thickness=torch.from_numpy(np.repeat(AREA, n)),
        fixed_nodes=torch.empty(0, dtype=torch.long), fixed_temperatures=torch.empty(0, dtype=torch.float64),
        contact_pairs=torch.from_numpy(np.column_stack((cuts, cuts+1))), contact_conductance=torch.from_numpy(1/resistance),
        convection_edges=torch.tensor([[0, 0], [last, last]]), convection_coefficient=0., ambient_temperature=0.,
        wall_areal_heat_capacity=torch.from_numpy(WALL_C))
    ends, drops = [], []
    continuity = contact_error = 0.
    drive_count = 4 if probe_contacts else 2
    for drive in np.eye(4)[:drive_count]:
        loads = np.zeros_like(x)
        loads[[0, last]] = drive[:2]
        loads[cuts], loads[cuts+1] = drive[2:], -drive[2:]
        driven = replace(model, nodal_heat=torch.from_numpy(loads))
        state = thermal.solve_harmonic_thermal(driven, omega, sparse=True).temperature
        local = thermal.thermal_element_nodal_power(driven, state, omega).numpy()
        t = state.numpy()
        jump = t[cuts]-t[cuts+1]
        # Nodal powers point into each element. The +P/-P contact probe
        # supplies the extra power crossing the resistor relative to the material.
        left = -local[[n-1, 2*n-1], 1]
        right = local[[n, 2*n], 0]
        continuity = max(continuity, abs(left-right).max())
        contact_error = max(contact_error, abs(left+drive[2:]-jump/resistance).max(),
                            abs(right+drive[2:]-jump/resistance).max())
        ends.append(t[[0, last]])
        drops.append(jump)
    diagnostics = dict(element_power_discontinuity_W=float(continuity),
                       element_contact_law_residual_W=float(contact_error))
    return np.column_stack(ends), np.column_stack(drops), diagnostics


def reference(omega, resistance):
    # Transfer temperature and rightward power through each layer and jump.
    transfer = np.eye(2)
    contact_rows = []
    for i, (length, area, k, c) in enumerate(zip(LENGTH, AREA, K, C)):
        transfer = expm(length*np.array([[0., -1/(area*k)], [-1j*omega*area*c, 0.]])) @ transfer
        if i < len(resistance):
            contact_rows.append(transfer[1].copy())
            transfer = np.array([[1., -resistance[i]], [0., 1.]]) @ transfer
    storage = 1j*omega*AREA[[0, -1]]*WALL_C
    initial = np.linalg.solve(np.vstack(([storage[0], 1.], [storage[1], -1.] @ transfer)), np.eye(2))
    return np.vstack((initial[0], (transfer @ initial)[0])), np.asarray(contact_rows) @ initial


def cases(n, omega, fem=False):
    solve = fem_response if fem else response
    ends, drops, *baseline = solve(n, omega, BASE, probe_contacts=True)
    q0 = drops[:, :2]/BASE[:, None]
    # Applying a series temperature jump at a contact sends power through
    # both contacts. This two-by-two map retains the mutual response.
    feedback = np.diag(1/BASE)-drops[:, 2:]/np.outer(BASE, BASE)
    for ratios in ([.1, 1.], [1., 10.], [.1, 10.], [10., .1], [4., 4.]):
        resistance = BASE*ratios
        delta = resistance-BASE
        q = np.linalg.solve(np.eye(2)+feedback*delta, q0)
        z = ends[:, :2]+q0.T @ (delta[:, None]*q)
        direct_z, direct_drops, *direct = solve(n, omega, resistance)
        diagnostics = {key: max(baseline[0][key], value) for key, value in direct[0].items()} if fem else {}
        exact_z, exact_q = reference(omega, resistance)
        yield dict(**diagnostics, cells_per_material=n, omega_rad_s=omega, resistance_K_W=resistance.tolist(),
                   impedance_error_K_W=float(abs(z-exact_z).max()),
                   contact_power_error_W=float(abs(q-exact_q).max()),
                   direct_impedance_difference_K_W=float(abs(z-direct_z).max()),
                   direct_power_difference_W=float(abs(q-direct_drops/resistance[:, None]).max()),
                   power_change_W=np.stack(((q-q0).real, (q-q0).imag), axis=-1).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem", action="store_true", help="verify the sparse LINE2 solver including both material-side element powers")
    fem = parser.parse_args().fem
    rows = []
    for omega in (.5, 4., 12.):
        previous = {}
        for n in (4, 8, 16, 32, 64):
            for row in cases(n, omega, fem):
                key = tuple(row["resistance_K_W"])
                errors = np.array([row["impedance_error_K_W"], row["contact_power_error_W"]])
                orders = np.log2(previous[key]/errors).tolist() if key in previous else []
                previous[key] = errors
                row.update(orders=orders, passed=bool(all(p > 1.8 for p in orders)
                           and (n < 64 or max(errors) < 2e-5)
                           and row["direct_impedance_difference_K_W"] < 1e-10
                           and row["direct_power_difference_W"] < 1e-10))
                if fem:
                    row["passed"] &= max(row["element_power_discontinuity_W"], row["element_contact_law_residual_W"]) < 1e-10
                rows.append(row)
    if fem:
        from tensorfem import thermal
        path, option, module = "../TensorFEM/src", " --fem", thermal
    else:
        from tensorfvm.cht import rod
        path, option, module = "../TensorFVM/src", "", rod
    print(json.dumps(dict(baseline_resistance_K_W=BASE.tolist(),
                          power_change_layout="contact x end drive x [real, imaginary], relative to baseline",
                          reference="Three continuum layer propagators and two contact temperature jumps, with storing end walls",
                          reproduction=f"OPENBLAS_NUM_THREADS=1 PYTHONPATH={path} python examples/harmonic_two_contact_updates.py{option}",
                          sources=[provenance(f) for f in (__file__, module.__file__)], runs=rows),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
