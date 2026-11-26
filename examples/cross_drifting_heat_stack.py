"""Constant-load heating of a three-layer slab with two thermal contacts."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod
from tensordg import heat_1d
from conduction_provenance import provenance


def measure(n, drive, resistance, storing=False, heated=False):
    length = np.array([.23, .46, .31])
    k = np.array([1.2, 3.4, .7])
    c = np.array([2.1, .9, 1.4])
    # Manufacture independent layer flux slopes and impose the jump at each interface.
    slopes = np.array([-.6, 1.3, -.9])
    left_capacity = np.array([.2, .65]) if storing else np.zeros(2)
    right_capacity = np.array([.8, .15]) if storing else np.zeros(2)
    left_load = np.array([.48, -.31]) if heated else np.zeros(2)
    right_load = np.array([-.48, .17]) if heated else np.zeros(2)
    left_t, left_q = .8, -.23
    cells, faces, traces = [], [], []
    for i, width in enumerate(length):
        x = np.linspace(0., width, n+1)
        q = left_q+slopes[i]*x
        t = left_t-(left_q*x+slopes[i]*x*x/2)/k[i]
        cells.append((t[:-1]+t[1:])/2+slopes[i]*(width/n)**2/(12*k[i]))
        faces.append(q)
        traces.append(t[[0, -1]])
        if i < len(resistance):
            through = q[-1]+left_load[i]-left_capacity[i]*drive
            left_t = t[-1]-resistance[i]*through
            left_q = through+right_load[i]-right_capacity[i]*drive
    mass = c*length
    traces = np.array(traces)
    shift = (mass@np.mean(cells, axis=1)+left_capacity@traces[:-1, 1]+right_capacity@traces[1:, 0])/(
        mass.sum()+left_capacity.sum()+right_capacity.sum())
    ends = (faces[0][0], -faces[-1][-1])
    source = c*drive+slopes
    _, modes, q, t, rate = heat_1d.solve_drifting_heat_stack_1d(
        length, k, c, ends, source=source, contact_resistance=resistance, ncells=n,
        interface_left_heat_sources=left_load, interface_heat_sources=right_load,
        interface_left_areal_heat_capacity=left_capacity, interface_areal_heat_capacity=right_capacity)
    dx = np.repeat(length/n, n)
    contacts = np.zeros(len(dx)-1)
    cuts = [n-1, 2*n-1]
    contacts[cuts] = resistance
    sheet = np.zeros_like(contacts)
    sheet[cuts] = left_capacity+right_capacity
    fraction = np.zeros_like(sheet)
    if storing:
        fraction[cuts] = left_capacity/sheet[cuts]
    trace_loads = np.zeros((2, contacts.size))
    trace_loads[:, cuts] = np.stack((left_load, right_load))
    u, outer, west, east, (before, after), fv_rate = rod.solve_harmonic_stepped_rod(
        dx, 1., np.repeat(k, n), np.repeat(c, n), ends, 0., source=np.repeat(source, n),
        end_types=("power", "power"), contact_resistance=contacts, return_mean_rate=True,
        trace_power=trace_loads, interface_heat_capacity=sheet, interface_capacity_left_fraction=fraction,
        flux_order=2, return_traces=True)
    fv_traces = np.column_stack((np.r_[outer[0], after[cuts]], np.r_[before[cuts], outer[-1]]))
    for solver, temp, powers, trace, heating in (
        ("DG", modes[:, :, 0].numpy(), q.numpy(), t.numpy(), rate),
        ("FVM", u.reshape(3, n), np.column_stack((west.reshape(3, n), east[n-1::n])), fv_traces, fv_rate),
    ):
        errors = dict(cell_mean_K=float(abs(temp-(np.asarray(cells)-shift)).max()),
                      face_flux_W_m2=float(abs(powers-faces).max()),
                      trace_K=float(abs(trace-(np.asarray(traces)-shift)).max()),
                      rate_K_s=float(abs(heating-drive)))
        errors["contact_balance_W_m2"] = float(abs(powers[:-1, -1]-powers[1:, 0]+left_load+right_load
                                                     -(left_capacity+right_capacity)*heating).max())
        through = powers[:-1, -1]+left_load-left_capacity*heating
        errors["jump_residual_K"] = float(abs(trace[:-1, 1]-trace[1:, 0]-resistance*through).max())
        assert max(errors.values()) < 1e-10, errors
        yield dict(solver=solver, cells_per_layer=n, rate_K_s=drive, resistance=resistance,
                   left_source_W_m2=left_load.tolist(), right_source_W_m2=right_load.tolist(),
                   left_capacity_J_m2_K=left_capacity.tolist(), right_capacity_J_m2_K=right_capacity.tolist(), errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-sources", action="store_true", help="apply signed loads to both contact traces")
    heated = parser.parse_args().contact_sources
    rows = [row for n in (3, 8, 16) for rate in (-.5, 0., .75)
            for resistance in ([0., .32], [.13, .32]) for storing in (False, True)
            for row in measure(n, rate, resistance, storing, heated)]
    print(json.dumps(dict(runs=rows, sources=[provenance(p) for p in (__file__, heat_1d.__file__, rod.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG:../TensorFVM/src python examples/cross_drifting_heat_stack.py"+(" --contact-sources" if heated else "")),
                     indent=2, allow_nan=False))
