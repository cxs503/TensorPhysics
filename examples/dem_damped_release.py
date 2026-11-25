"""Verify force release and geometric separation for a clamped linear dashpot."""
import json
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from conduction_provenance import provenance
from tensordem import contact_impact, dem3d


def measure(zeta, scale):
    stiffness, density, radius = 5e4, 900.0, 0.05
    reduced_mass = density * 2 * math.pi * radius**3 / 3
    frequency = math.sqrt(stiffness / reduced_mass)
    matrix = np.array([[0.0, 1.0], [-1.0, -2 * zeta]])

    def state(t):
        return expm(t * matrix)[:, 1]

    def trial_force(t):
        displacement, speed = state(t)
        return displacement + 2 * zeta * speed

    release = brentq(trial_force, 0.01, 4.0)
    displacement, velocity = state(release)
    separation = release - displacement / velocity
    restitution = 1.0 if zeta == 0 else -velocity
    result = contact_impact.pair_impact(
        e_target=restitution, damping="clamped_mapping", v0=0.5, k_n=stiffness,
        density=density, radius=radius, dt_scale=scale,
    )
    force = result.time_to_force_release_s
    gap = result.time_to_separation_s
    release_error = abs(force * frequency / release - 1)
    separation_error = abs(gap * frequency / separation - 1)
    interval_error = abs((gap - force) * frequency - (separation - release)) / separation
    return dict(
        zeta=zeta, dt_scale=scale, dt_s=result.dt, release_s=force, separation_s=gap,
        exact_release_s=release / frequency, exact_separation_s=separation / frequency,
        force_free_interval_s=gap - force,
        exact_force_free_interval_s=(separation - release) / frequency,
        relative_release_error=release_error, relative_separation_error=separation_error,
        normalized_force_free_error=interval_error,
        passed=bool(force > 0 and gap >= force - 1e-14 and
                    max(release_error, separation_error) < 0.008 * scale and
                    interval_error < 3e-6),
    )


if __name__ == "__main__":
    rows = [measure(zeta, scale) for zeta in (0.0, 0.35, 1.0, 1.6)
            for scale in (0.25, 0.125, 0.0625)]
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(
        reference="First zero of the unloading force from a two-state matrix exponential, followed by constant-speed separation",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_damped_release.py",
        runs=rows, passed=passed,
        sources=[provenance(p) for p in (__file__, contact_impact.__file__, dem3d.__file__)],
    ), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
