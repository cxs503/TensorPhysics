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
    start = angle/omega if omega else 0.
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


def steady_drift(n, balanced):
    # Linear conductance law and nodal conservation give each link's current
    # by a prefix sum, independently of the solver's constrained matrix.
    r = np.geomspace(.07, 1.8, n-1)
    capacity = np.linspace(.3, 2.4, n)
    capacity[1::3] = 0.
    p = np.cos(np.arange(n)*.7)+1j*np.sin(np.arange(n)*.41)
    if balanced:
        p[-1] = -p[:-1].sum()
    rate = p.sum()/capacity.sum()
    q = np.cumsum(p-capacity*rate)[:-1]
    exact = np.r_[0., -np.cumsum(r*q)]
    exact -= capacity@exact/capacity.sum()
    t, slope, drift = contact_chain.low_frequency_contact_temperatures(r, capacity, nodal_power=p,
                                                                    allow_massless=True)
    slope_current = -np.cumsum(capacity*exact)[:-1]
    slope_exact = np.r_[0., -np.cumsum(r*slope_current)]
    slope_exact -= capacity@slope_exact/capacity.sum()
    slope_q = -np.diff(slope)/r
    expansion = []
    previous = None
    for scaled_frequency in (.2, .1, .05, .025):
        omega = scaled_frequency/(capacity.sum()*r.sum())
        field, _ = contact_chain.harmonic_contact_temperatures(r, capacity, omega, nodal_power=p,
                                                              allow_massless=True, separate_mean=True)
        error = t+1j*omega*slope-field
        remainder = np.array([abs(error).max(), abs(np.diff(error)/r).max()])
        orders = np.log2(previous/remainder).tolist() if previous is not None else []
        expansion.append(dict(omega_rad_s=omega, temperature_remainder_K=float(remainder[0]),
                              link_power_remainder_W=float(remainder[1]), temperature_and_power_orders=orders))
        previous = remainder
    return dict(nodes=n, balanced_power=balanced, expansion=expansion,
                temperature_slope_relative_error=float(abs(slope-slope_exact).max()/abs(slope_exact).max()),
                power_slope_relative_error=float(abs(slope_q-slope_current).max()/abs(slope_current).max()), relative_temperature_error_K=float(abs(t-exact).max()),
                contact_power_error_W=float(abs(-np.diff(t)/r-q).max()),
                mean_heating_rate_error_K_s=float(abs(drift-rate)),
                centered_mean_error_K=float(abs(capacity@t)/capacity.sum()))


if __name__ == "__main__":
    rows = [measure(w, dt, phase) for w in (4., 1e-4, 1e-12, 1e-16)
            for dt in (.023, .73) for phase in (0., .83)]
    rows.extend(measure(0., dt, 0.) for dt in (.023, .73))
    static = [steady_drift(n, balanced) for n in (3, 9, 33) for balanced in (False, True)]
    for row in static:
        row["passed"] = all(value < 1e-12 for key, value in row.items() if "_error_" in key or key.endswith("_relative_error"))
        row["passed"] &= all(1.95 < order < 2.05 for sample in row["expansion"]
                             for order in sample["temperature_and_power_orders"])
    for row in rows:
        row["passed"] = max(row[k] for k in ("endpoint_error_K", "mean_error_K", "link_heat_error_J",
                                             "node_energy_residual_J")) < 1e-12
    print(json.dumps(dict(reference="Independent real-time modal ODE with a driven algebraic contact",
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_insulated_contact_harmonic.py",
        runs=rows, zero_frequency_profiles=static, sources=[provenance(__file__), provenance(contact_chain.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows+static) else 1)
