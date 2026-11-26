"""Run with python examples/equal_wall_ramp_and_hold.py from TensorPhysics.

The three solver packages must be importable from their Git checkouts.
Compare inward heat per unit area for a homogeneous unit slab, initially at
zero, with unit conductivity and volumetric heat capacity.
"""
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


def fem(n, times):
    x = torch.linspace(0, 1, n+1, dtype=torch.float64)
    conn = torch.stack((torch.arange(n), torch.arange(1, n+1)), dim=1)
    ends = torch.tensor([0, n])
    model = thermal.ThermalModel(x[:, None], conn, 1., 1., 1., ends, x.new_zeros(2))
    values = x.new_tensor(wall(times))[:, None].expand(-1, 2)
    result = thermal.solve_transient_thermal(model, x.new_tensor(times), x.new_zeros(n+1),
                                             theta=.5, fixed_temperature_history=values)
    return (torch.trapezoid(result.temperature, x).numpy(),
            result.nodal_reaction_heat_steps()[:, ends].numpy())


def fvm(n, times):
    boundary = dict(type="dirichlet", values=wall)
    states, heat = energy.solve_transient_conduction(
        np.full(n, 1/n), [1.], 1., 1., {"west": boundary, "east": boundary},
        times, 0., theta=.5, return_face_heat=True)
    return states[:, 0].mean(axis=1), np.column_stack([heat["wall"][s][:, 0] for s in ("west", "east")])


def particles(n, times):
    states, heat = sph.solve_transient_layered_slab(
        np.zeros(n), times, 1/n, (wall, wall), conductivity=np.ones(n),
        volumetric_heat_capacity=1., method="exponential", return_x_heat=True)
    return states.numpy().mean(axis=1), heat[:, [0, -1]].numpy()*[1., -1.]


def run():
    reference = reference_heat()
    rows = []
    for name, solve in (("TensorFEM", fem), ("TensorFVM", fvm), ("TensorSPH", particles)):
        previous = None
        for n, steps in ((12, 32), (24, 64), (48, 256)):
            # Exponential SPH integrates the affine load on each phase directly.
            steps = 1 if name == "TensorSPH" else steps
            times = np.r_[np.linspace(*TIMES[:2], steps+1), np.linspace(*TIMES[1:], steps+1)[1:]]
            storage, inward = solve(n, times)
            inputs = inward.reshape(2, steps, 2).sum(axis=1)
            error = np.max(abs(inputs-reference[:, None]), axis=1)
            balance = np.max(abs(np.diff(storage[::steps])-inputs.sum(axis=1)))
            passed = (balance < 1e-12 and (previous is None or np.all(previous > 3*error))
                      and (n < 48 or np.all(error < .01*reference)))
            rows.append(dict(solver=name, resolution=n, steps_per_phase=steps,
                             wall_inputs_J_m2=inputs.tolist(), max_wall_errors_J_m2=error.tolist(),
                             max_balance_J_m2=float(balance), passed=bool(passed)))
            previous = error
    return dict(times_s=TIMES.tolist(), phases=["ramp", "hold"], walls=["west", "east"],
                reference_per_wall_J_m2=reference.tolist(), runs=rows,
                sources=[provenance(p) for p in (__file__, thermal.__file__, energy.__file__, sph.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
