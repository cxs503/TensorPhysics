"""Compare an elastic sphere/platen contact trajectory with the inverse Hertz time integral."""
import json
import math

import numpy as np
from scipy.special import beta, betaincinv

from conduction_provenance import provenance
from tensordem import dem3d, hertz_impact_time as impact


def measure(speed, steps):
    radius, density, young, nu = .025, 917., 1e9, .3
    mass = density*4*math.pi*radius**3/3
    stiffness = 4*young*math.sqrt(radius)/(3*(1-nu**2))
    peak = (5*mass*speed**2/(4*stiffness))**.4
    duration = .8*beta(.4,.5)*peak/speed
    dt = duration/steps
    result = impact.measure_platen_impact(v_in=speed,dt=dt,steps_per_contact=steps,
        radius=radius,density=density,young_modulus=young,poisson_ratio=nu,record_history=True)
    time, overlap, velocity = np.asarray(result["history"]).T
    force_time, reaction = np.asarray(result["platen_reaction_history"]).T

    def reference(t):
        fraction = np.clip(t/duration,0.,1.)
        # The normalized compression time is I_z(2/5, 1/2), z=(delta/peak)^(5/2).
        z = betaincinv(.4,.5,2*np.minimum(fraction,1-fraction))
        return peak*z**.4, np.where(fraction < .5,-1.,1.)*speed*np.sqrt(1-z)

    exact_overlap, exact_velocity = reference(time)
    force_overlap, _ = reference(force_time)
    exact_reaction = -stiffness*force_overlap**1.5
    # The platen reaction is evaluated before the velocity and position update.
    impulse = -np.cumsum(reaction)*dt
    momentum = mass*(velocity+speed)
    energy = .5*mass*velocity**2 + .4*stiffness*np.maximum(overlap,0.)**2.5
    return dict(speed_m_s=speed,steps_per_contact=steps,dt_s=dt,reference_duration_s=duration,
        relative_overlap_error=float(np.max(abs(np.maximum(overlap,0.)-exact_overlap))/peak),
        relative_velocity_error=float(np.max(abs(velocity-exact_velocity))/speed),
        relative_reaction_error=float(np.max(abs(reaction-exact_reaction))/(stiffness*peak**1.5)),
        relative_energy_drift=float(np.max(abs(energy/(.5*mass*speed**2)-1))),
        relative_impulse_balance=float(np.max(abs(impulse-momentum))/(mass*speed)),
        reference_impulse_Ns=2*mass*speed,measured_impulse_Ns=float(impulse[-1]))


if __name__ == "__main__":
    rows = []
    metrics = ["relative_overlap_error","relative_velocity_error","relative_reaction_error","relative_energy_drift"]
    for speed in (.2,.8):
        previous = None
        for steps in (256,512,1024):
            row = measure(speed,steps)
            errors = np.array([row[key] for key in metrics])
            row["orders"] = [] if previous is None else np.log2(previous/errors).tolist()
            row["passed"] = bool(row["relative_impulse_balance"] < 1e-8
                and (previous is None or np.all(np.array(row["orders"]) > [1.9,.95,1.9,.95]))
                and (steps < 1024 or np.all(errors < [2e-6,.002,1.5e-6,.002])))
            rows.append(row)
            previous = errors
    print(json.dumps(dict(reference="F=k delta^(3/2), energy integral inverted via the regularized incomplete beta function on compression and release",
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_hertz_contact_history.py",
        order_metrics=metrics,runs=rows,sources=[provenance(p) for p in (__file__,impact.__file__,dem3d.__file__)]),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
