"""Closed two-grain network with a heated, zero-capacity contact between the stores."""
import json

import numpy as np
from scipy.linalg import expm

from conduction_provenance import provenance
from tensordem.thermal import contact_chain


def measure(omega, duration, angle):
    resistances = np.array([.23, .47])
    r1, r2 = resistances
    r = r1+r2
    c1, c2 = 2., 3.
    powers = np.array([.8+.1j, .27-.32j, -.3+.2j])
    p1, pc, p2 = powers
    initial = [.4, 0., -.6]
    start = angle/omega
    final, heat, mean = contact_chain.harmonic_contact_interval(
        resistances, [c1, 0., c2], omega, start, duration, initial, t_left=None, t_right=None,
        nodal_power=powers, allow_massless=True, return_interval_temperature=True)

    # In physical time the capacity mean and left-minus-right difference
    # satisfy two scalar ODEs. Append the drive oscillator and its integrals
    # so one small real exponential gives both endpoint and interval averages.
    rate = powers.sum()/(c1+c2)
    diff_drive = (p1+pc*r2/r)/c1-(p2+pc*r1/r)/c2
    a = np.zeros((8, 8))
    a[0, 2:4] = rate.real, -rate.imag
    a[1, 1] = -(1/c1+1/c2)/r
    a[1, 2:4] = diff_drive.real, -diff_drive.imag
    a[2, 3], a[3, 2] = -omega, omega
    a[4:, :4] = np.eye(4)
    u = np.array([(c1*initial[0]+c2*initial[2])/(c1+c2), initial[0]-initial[2],
                  np.cos(angle), np.sin(angle), 0., 0., 0., 0.])
    state = expm(duration*a)@u

    def nodes(values):
        common, difference, cosine, sine = values
        left = common+c2*difference/(c1+c2)
        right = common-c1*difference/(c1+c2)
        return np.array([left, (r2*left+r1*right+r1*r2*(pc.real*cosine-pc.imag*sine))/r, right])

    target = nodes(state[:4])
    avg = nodes(state[4:])/duration
    energy = (powers*(state[6]+1j*state[7])).real
    balance = (final-initial)*[c1, 0., c2]+np.diff(np.r_[0.,heat,0.])-energy
    return dict(omega_rad_s=omega, duration_s=duration, start_phase_rad=angle,
                endpoint_error_K=float(abs(final-target).max()), mean_error_K=float(abs(mean-avg).max()),
                link_heat_error_J=float(abs(heat+duration*np.diff(avg)/resistances).max()),
                node_energy_residual_J=float(abs(balance).max()))


if __name__ == "__main__":
    rows = [measure(w, dt, phase) for w in (4., 1e-4, 1e-12, 1e-16)
            for dt in (.023, .73) for phase in (0., .83)]
    for row in rows:
        row["passed"] = max(row[k] for k in ("endpoint_error_K", "mean_error_K", "link_heat_error_J",
                                             "node_energy_residual_J")) < 1e-12
    print(json.dumps(dict(reference="Independent real-time modal ODE with a driven algebraic contact",
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_insulated_contact_harmonic.py",
        runs=rows, sources=[provenance(__file__), provenance(contact_chain.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
