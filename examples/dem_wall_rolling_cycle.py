"""Verify rolling hysteresis against a closed-form elastic/perfectly sliding cycle."""
import json
import math

import numpy as np

from conduction_provenance import provenance
from tensordem import granular3d_core
from tensordem.granular_column import kernel, pluviation


def measure(amplitude_ratio, steps_per_leg, direction):
    material = granular3d_core.Material3D(radius=.5, kn=100., kt_ratio=.5, mu=0.)
    state = kernel.WallSpringState(1)
    # Constant 0.25 m penetration, 25 N normal force, 0.05 rolling coefficient.
    stiffness = 12.5
    torque_cap = .625
    yield_angle = torque_cap / stiffness
    amplitude = amplitude_ratio * yield_angle
    turning_extension = min(amplitude, yield_angle)
    dt = 1. / steps_per_leg
    previous_angle = 0.
    accumulated_loss = 0.
    torque_error = energy_error = loss_error = 0.

    # First establish the negative turning point. Measure a full periodic
    # excursion from there, excluding the initial loading work.
    for leg, (start, end) in enumerate(((0., -amplitude), (-amplitude, amplitude),
                                        (amplitude, -amplitude))):
        if leg == 1:
            accumulated_loss = 0.
        for i in range(1, steps_per_leg + 1):
            fraction = i / steps_per_leg
            angle = start + (end - start) * fraction
            azimuth = direction * math.pi * (leg + fraction)
            radial = np.array([math.cos(azimuth), math.sin(azimuth), 0.])
            axis = .6 * np.array([-radial[1], radial[0], 0.]) + [0., 0., .8]
            spin = (angle - previous_angle) / dt * axis
            previous_angle = angle
            result = pluviation.cylinder_wall_contact_forces(
                .75 * radial[None, :], np.zeros((1, 3)), spin[None, :], material, dt,
                wall_radius=1., mu_rolling_wall=.05, state=state, corotate_wall_rolling=True,
            )
            accumulated_loss += result["dissip_rolling"]
            if leg == 0:
                extension = max(-yield_angle, angle)
                exact_loss = torque_cap * max(0., -angle - yield_angle)
            else:
                progress = abs(angle - start)
                sign = 1 if leg == 1 else -1
                extension = sign * min(yield_angle, progress - turning_extension)
                sliding = max(0., progress - (yield_angle + turning_extension))
                if leg == 2:
                    sliding += 2. * max(0., amplitude - yield_angle)
                exact_loss = torque_cap * sliding
            torque_error = max(torque_error, np.linalg.norm(result["torque"][0] + stiffness * extension * axis))
            energy_error = max(energy_error, abs(result["energy_rolling"] - .5 * stiffness * extension**2))
            loss_error = max(loss_error, abs(accumulated_loss - exact_loss))

    return dict(amplitude_over_yield=amplitude_ratio, steps_per_leg=steps_per_leg,
                azimuth_direction=direction, measured_cycle_loss_J=accumulated_loss,
                reference_cycle_loss_J=4. * torque_cap * max(0., amplitude - yield_angle),
                max_torque_error_Nm=float(torque_error), max_energy_error_J=energy_error,
                max_cumulative_loss_error_J=loss_error,
                passed=bool(max(torque_error/torque_cap, energy_error/(torque_cap*yield_angle),
                                loss_error/(torque_cap*yield_angle)) < 1e-11))


if __name__ == "__main__":
    rows = [measure(amplitude, steps, direction) for amplitude in (.6, 1., 2.7)
            for steps in (7, 31) for direction in (-1, 1)]
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(
        reference="Periodic elastic-perfectly sliding rolling spring: cycle loss = 4 M_cap max(A - M_cap/k, 0)",
        reproduction="PYTHONPATH=../TensorDEM/src python examples/dem_wall_rolling_cycle.py",
        runs=rows, passed=passed,
        sources=[provenance(p) for p in (__file__, granular3d_core.__file__, kernel.__file__, pluviation.__file__)],
    ), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
