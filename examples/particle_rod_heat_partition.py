"""Verify a DEM pair and a two-cell FVM rod against the same thermal network ODE.

Run python examples/particle_rod_heat_partition.py with both solvers installed.
The comparison concerns the lumped network and time integration. Each node
has C=1 J/K, contact G=1 W/K and ambient conductance h=1 W/K.
Temperatures are offsets from a common 300 K datum.
"""
import json

import numpy as np
from tensordem.interface_thermal import conduction
from tensorfvm.cht import rod

from conduction_provenance import provenance


INITIAL = np.array([-1., 1.])
AMBIENT = np.array([2., 0.])
POWER = np.array([1., -1.])


def dem(times):
    # Unit patch radius. Two Maxwell resistances plus the CMY film give R=1.
    k = .5 + 1/(1.25*np.pi)
    particle = conduction.Particle(2., k, 3/(32*np.pi), 1.)
    pair = conduction.ParticleThermalNetwork(
        [particle, particle], [(0, 1)], patch_radius_fraction=.5, rms_roughness_m=1.,
        rms_slope=1., contact_pressure_pa=1., hardness_pa=1.)
    pair.reset(INITIAL)
    fields, contacts, ambient = [INITIAL], [], []
    for dt in np.diff(times):
        field, link, air = pair.step(dt, ambient=AMBIENT, h_ambient_w_k=[1., 1.],
                                    particle_power_w=POWER, return_contact_heat=True,
                                    return_ambient_heat=True)
        fields.append(field)
        contacts.append(link[0])
        ambient.append(air)
    return np.array(fields), np.array(contacts), np.array(ambient)


def fvm(times):
    # Unit cell volumes; the two half-cell resistances sum to 0.5 K/W.
    field, _, _, _, heat = rod.solve_transient_stepped_rod(
        [1., 1.], 1., 2., 1., [0., 0.], times, INITIAL, theta=.5,
        end_types=("power", "power"), contact_resistance=.5, source=POWER,
        heat_loss_coefficient=1., bulk_ambient_temperature=AMBIENT, return_step_heat=True)
    return field, heat["contact"][:, 0], heat["bulk_exchange"]


def reference(t):
    # Mean m and half-difference d obey m'=1-m and d'=2-3d.
    mean = -np.expm1(-t)
    diff = 2/3 - (5/3)*np.exp(-3*t)
    int_mean = t - mean
    int_diff = 2*t/3 + (5/9)*np.expm1(-3*t)
    return (np.array([mean+diff, mean-diff]), 2*int_diff,
            t*AMBIENT - [int_mean+int_diff, int_mean-int_diff])


def run():
    end = 2.
    target, contact, ambient = reference(end)
    reversal = np.log(2.5)/3
    rows = []
    for name, solve, minimum_ratio in (("TensorDEM_backward_Euler", dem, 1.8),
                                       ("TensorFVM_Crank_Nicolson", fvm, 3.8)):
        previous = None
        for steps in (16, 32, 64, 128):
            times = np.linspace(0., end, steps+1)
            field, link, air = solve(times)
            residual = np.diff(field, axis=0) - np.diff(times)[:, None]*POWER - air
            residual += np.column_stack((link, -link))
            balance = np.max(abs(residual))
            errors = np.array([max(abs(field[-1]-target)), abs(link.sum()-contact),
                               max(abs(air.sum(axis=0)-ambient))])
            switches = np.flatnonzero(np.diff(np.signbit(link))) + 1
            passed = (balance < 1e-13 and link[0] < 0 < link[-1] and len(switches) == 1
                      and abs(times[switches[0]]-reversal) < end/steps)
            passed &= previous is None or np.all(previous > minimum_ratio*errors)
            passed &= steps < 128 or max(errors) < (.005 if solve is dem else 2e-5)
            rows.append(dict(solver=name, steps=steps, temperature_error_K=errors[0],
                             contact_error_J=errors[1], max_ambient_error_J=errors[2],
                             max_nodal_balance_J=balance, passed=bool(passed)))
            previous = errors
    return dict(command="python examples/particle_rod_heat_partition.py", end_s=end,
                reference_reversal_s=reversal, reference_final_temperature_offsets_K=target.tolist(), reference_contact_J=contact,
                reference_ambient_J=ambient.tolist(), runs=rows,
                sources=[provenance(p) for p in (__file__, conduction.__file__, rod.__file__)])


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in report["runs"]) else 1)
