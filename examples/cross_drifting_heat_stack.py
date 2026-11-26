"""Constant-load heating of a three-layer slab with two thermal contacts."""
import argparse
import json

import numpy as np
from tensorfvm.cht import rod
from tensordg import heat_1d
from conduction_provenance import provenance


def measure(n, drive, resistance, storing=False, heated=False, stepped=False, shoulders=False):
    length = np.array([.23, .46, .31])
    area = np.array([.65, 1.7, .95]) if stepped else np.ones(3)
    overlap = np.minimum(area[:-1], area[1:])
    k = np.array([1.2, 3.4, .7])
    c = np.array([2.1, .9, 1.4])
    # Manufacture independent layer flux slopes and impose the jump at each interface.
    slopes = np.array([-.6, 1.3, -.9])
    left_capacity = np.array([.2, .65]) if storing else np.zeros(2)
    right_capacity = np.array([.8, .15]) if storing else np.zeros(2)
    shoulder_capacity = np.array([.7, 1.3]) if shoulders else np.zeros(2)
    area_step = np.diff(area)
    left_mass = left_capacity*overlap+shoulder_capacity*np.maximum(-area_step, 0.)
    right_mass = right_capacity*overlap+shoulder_capacity*np.maximum(area_step, 0.)
    left_load = np.array([.48, -.31]) if heated else np.zeros(2)
    right_load = np.array([-.48, .17]) if heated else np.zeros(2)
    left_t, left_q = .8, -.23
    cells, faces, traces = [], [], []
    for i, width in enumerate(length):
        x = np.linspace(0., width, n+1)
        q = left_q+slopes[i]*x
        t = left_t-(left_q*x+slopes[i]*x*x/2)/(k[i]*area[i])
        cells.append((t[:-1]+t[1:])/2+slopes[i]*(width/n)**2/(12*k[i]*area[i]))
        faces.append(q)
        traces.append(t[[0, -1]])
        if i < len(resistance):
            through = q[-1]/overlap[i]+left_load[i]-left_mass[i]*drive/overlap[i]
            left_t = t[-1]-resistance[i]*through
            left_q = overlap[i]*(through+right_load[i]-right_mass[i]*drive/overlap[i])
    mass = c*length*area
    traces = np.array(traces)
    shift = (mass@np.mean(cells, axis=1)+left_mass@traces[:-1, 1]+right_mass@traces[1:, 0])/(
        mass.sum()+left_mass.sum()+right_mass.sum())
    ends = (faces[0][0], -faces[-1][-1])
    source = c*drive+slopes/area
    _, modes, q, t, rate = heat_1d.solve_drifting_heat_stack_1d(
        length, k, c, ends/area[[0,-1]], area=area, source=source, contact_resistance=resistance, ncells=n,
        shoulder_areal_heat_capacity=shoulder_capacity,
        interface_left_heat_sources=left_load, interface_heat_sources=right_load,
        interface_left_areal_heat_capacity=left_capacity, interface_areal_heat_capacity=right_capacity)
    dx = np.repeat(length/n, n)
    contacts = np.zeros(len(dx)-1)
    cuts = [n-1, 2*n-1]
    contacts[cuts] = resistance
    sheet = np.zeros_like(contacts)
    sheet[cuts] = (left_capacity+right_capacity)*overlap
    shoulder_profile = np.zeros_like(contacts)
    shoulder_profile[cuts] = shoulder_capacity
    fraction = np.zeros_like(sheet)
    if storing:
        fraction[cuts] = left_capacity*overlap/sheet[cuts]
    trace_loads = np.zeros((2, contacts.size))
    trace_loads[:, cuts] = np.stack((left_load, right_load))*overlap
    u, outer, west, east, (before, after), fv_rate = rod.solve_harmonic_stepped_rod(
        dx, np.repeat(area, n), np.repeat(k, n), np.repeat(c, n), ends, 0., source=np.repeat(source, n),
        end_types=("power", "power"), contact_resistance=contacts, return_mean_rate=True,
        shoulder_areal_heat_capacity=shoulder_profile,
        trace_power=trace_loads, interface_heat_capacity=sheet, interface_capacity_left_fraction=fraction,
        flux_order=2, return_traces=True)
    fv_traces = np.column_stack((np.r_[outer[0], after[cuts]], np.r_[before[cuts], outer[-1]]))
    for solver, temp, powers, trace, heating in (
        ("DG", modes[:, :, 0].numpy(), q.numpy()*area[:, None], t.numpy(), rate),
        ("FVM", u.reshape(3, n), np.column_stack((west.reshape(3, n), east[n-1::n])), fv_traces, fv_rate),
    ):
        errors = dict(cell_mean_K=float(abs(temp-(np.asarray(cells)-shift)).max()),
                      face_flux_W_m2=float(abs((powers-faces)/area[:, None]).max()),
                      trace_K=float(abs(trace-(np.asarray(traces)-shift)).max()),
                      rate_K_s=float(abs(heating-drive)))
        errors["face_power_W"] = float(abs(powers-faces).max())
        errors["contact_balance_W_m2"] = float(abs((powers[:-1, -1]-powers[1:, 0])/overlap+left_load+right_load
                                                     -(left_mass+right_mass)*heating/overlap).max())
        through = powers[:-1, -1]/overlap+left_load-left_mass*heating/overlap
        errors["jump_residual_K"] = float(abs(trace[:-1, 1]-trace[1:, 0]-resistance*through).max())
        assert max(errors.values()) < 1e-10, errors
        yield dict(solver=solver, cells_per_layer=n, rate_K_s=drive, resistance=resistance, area_m2=area.tolist(),
                   shoulder_capacity_J_m2_K=shoulder_capacity.tolist(),
                   left_source_W_m2=left_load.tolist(), right_source_W_m2=right_load.tolist(),
                   left_capacity_J_m2_K=left_capacity.tolist(), right_capacity_J_m2_K=right_capacity.tolist(), errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-sources", action="store_true", help="apply signed loads to both contact traces")
    parser.add_argument("--stepped", action="store_true", help="expand then contract the layer cross section")
    parser.add_argument("--shoulder-storage", action="store_true", help="apply heat capacity to the exposed shoulders")
    args = parser.parse_args()
    heated = args.contact_sources
    rows = [row for n in (3, 8, 16) for rate in (-.5, 0., .75)
            for resistance in ([0., .32], [.13, .32]) for storing in (False, True)
            for row in measure(n, rate, resistance, storing, heated, args.stepped, args.shoulder_storage)]
    print(json.dumps(dict(runs=rows, sources=[provenance(p) for p in (__file__, heat_1d.__file__, rod.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG:../TensorFVM/src python examples/cross_drifting_heat_stack.py"+(" --contact-sources" if heated else "")+(" --stepped" if args.stepped else "")+(" --shoulder-storage" if args.shoulder_storage else "")),
                     indent=2, allow_nan=False))
