"""Run with python examples/equal_wall_ramp_and_hold.py from TensorPhysics.

The three solver packages must be importable from their Git checkouts.
Compare inward heat per unit area for a homogeneous unit slab, initially at
zero, with unit conductivity and volumetric heat capacity. Pass --layered for
two equal-width layers of matched effusivity and a rightward interface heat check.
"""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


TIMES = np.array([0., .04, .09])


def wall(t):
    return np.minimum(t/TIMES[1], 1.)


def reference_heat():
    wave = np.pi*np.arange(1, 80, 2)

    def storage(t):
        # Spatial integral of the odd sine series for an uninterrupted ramp.
        return (t-1/12+np.sum(8*np.exp(-wave**2*t)/wave**4))/TIMES[1]

    # Subtract a delayed ramp to keep the wall fixed after TIMES[1].
    return np.diff([0., storage(TIMES[1]), storage(TIMES[2])-storage(TIMES[2]-TIMES[1])])/2


def reference_interface_heat():
    # y = integral(dx/k). For k*C=1 the transformed slab has unit diffusivity.
    # In these coordinates the physical interface falls at y=1/4.
    y = .25
    wave = np.pi*np.arange(1, 80, 2)

    def cumulative(t):
        tail = np.sum(4*np.cos(wave*y)*np.exp(-wave**2*t)/wave**4)
        return (t*(.5-y)-1/24+y*y/4-y**3/6+tail)/TIMES[1]

    return np.diff([0., cumulative(TIMES[1]), cumulative(TIMES[2])-cumulative(TIMES[2]-TIMES[1])])


def fem(n, times, k):
    x = torch.linspace(0, 1, n+1, dtype=torch.float64)
    conn = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
    ends = torch.tensor([0, n])
    model = thermal.ThermalModel(x[:, None], conn, x.new_tensor(k), 1., x.new_tensor(1/k), ends, x.new_zeros(2))
    values = x.new_tensor(wall(times))[:, None].expand(-1, 2)
    result = thermal.solve_transient_thermal(model, x.new_tensor(times), x.new_zeros(n+1),
                                             theta=.5, fixed_temperature_history=values)
    cell_storage = result.temperature[:, conn].mean(dim=-1).numpy()/(n*k)
    inward = result.nodal_reaction_heat_steps()[:, ends].numpy()
    local = thermal.thermal_element_nodal_heat_steps(model, result)
    return cell_storage, inward, -local[:, n//2-1, 1].numpy()


def fvm(n, times, k):
    boundary = dict(type="dirichlet", values=wall)
    states, heat = energy.solve_transient_conduction(
        np.full(n, 1/n), [1.], k, 1/k, {"west": boundary, "east": boundary},
        times, 0., theta=.5, return_face_heat=True)
    inward = np.column_stack([heat["wall"][s][:, 0] for s in ("west", "east")])
    return states[:, 0]/(n*k), inward, heat["interior"]["x"][:, 0, n//2-1]


def particles(n, times, k):
    states, heat = sph.solve_transient_layered_slab(
        np.zeros(n), times, 1/n, (wall, wall), conductivity=k,
        volumetric_heat_capacity=1/k, method="exponential", return_x_heat=True)
    return states.numpy()/(n*k), heat[:, [0, -1]].numpy()*[1., -1.], heat[:, n//2].numpy()


def run(layered=False):
    # The transformed slab has unit total resistance and storage in both cases,
    # so the analytic wall heat is identical, even when the cut heat differs.
    reference = reference_heat()
    cut_reference = reference_interface_heat() if layered else np.zeros(2)
    rows = []
    for name, solve in (("TensorFEM", fem), ("TensorFVM", fvm), ("TensorSPH", particles)):
        previous = None
        previous_cut_error = None
        for n, steps in ((12, 32), (24, 64), (48, 256)):
            # Exponential SPH integrates the affine load on each phase directly.
            steps = 1 if name == "TensorSPH" else steps
            times = np.r_[np.linspace(*TIMES[:2], steps+1), np.linspace(*TIMES[1:], steps+1)[1:]]
            k = np.repeat([2., 2/3], n//2) if layered else np.ones(n)
            storage, inward, cut_steps = solve(n, times, k)
            cut = cut_steps.reshape(2, steps).sum(axis=1)
            inputs = inward.reshape(2, steps, 2).sum(axis=1)
            error = np.max(abs(inputs-reference[:, None]), axis=1)
            balance = np.max(abs(np.diff(storage[::steps].sum(axis=1))-inputs.sum(axis=1)))
            passed = (balance < 1e-12 and (previous is None or np.all(previous > 3*error))
                      and (n < 48 or np.all(error < .01*reference)))
            cut_error = abs(cut-cut_reference)
            if layered:
                stored = storage[::steps]
                gains = np.column_stack((np.diff(stored[:, :n//2].sum(axis=1)),
                                         np.diff(stored[:, n//2:].sum(axis=1))))
                net = inputs + np.column_stack((-cut, cut))
                layer_balance = float(np.max(abs(gains-net)))
                passed = (passed and layer_balance < 1e-12
                          and (previous_cut_error is None or np.all(previous_cut_error > 3*cut_error))
                          and (n < 48 or np.all(cut_error < .01*cut_reference)))
                previous_cut_error = cut_error
            rows.append(dict(solver=name, resolution=n, steps_per_phase=steps,
                             wall_inputs_J_m2=inputs.tolist(), max_wall_errors_J_m2=error.tolist(),
                             max_balance_J_m2=float(balance), passed=bool(passed)))
            if layered:
                rows[-1].update(interface_heat_J_m2=cut.tolist(), interface_error_J_m2=cut_error.tolist(),
                                max_layer_balance_J_m2=layer_balance)
            previous = error
    report = dict(times_s=TIMES.tolist(), phases=["ramp", "hold"], walls=["west", "east"],
                reference_per_wall_J_m2=reference.tolist(), runs=rows,
                sources=[provenance(p) for p in (__file__, thermal.__file__, energy.__file__, sph.__file__)])
    if layered:
        report.update(conductivity_W_mK=[2., 2/3], capacity_J_m3K=[.5, 1.5],
                      interface_x_m=.5, reference_rightward_interface_J_m2=cut_reference.tolist())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layered", action="store_true", help="verify two matched-effusivity layers")
    report = run(layered=parser.parse_args().layered)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
