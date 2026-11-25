"""Verify short-pulse heating across a resolved, massless resistive contact."""
import json
import math

import numpy as np
from scipy.optimize import brentq
from tensordem.thermal import contact_chain

import dg_reversing_wall_pulse as pulse
from conduction_provenance import provenance


def measure(n, resistance, width):
    # A symmetric sine has zero contact flux. The antisymmetric mode has
    # one-sided temperatures +/-sin(k/2) and rightward flux -k*cos(k/2).
    pi = math.pi
    k = brentq(lambda k: 2*math.sin(k/2) + resistance*k*math.cos(k/2), pi, 2*pi)
    rates = np.square([pi, k])
    dt = 1/(16*pi*pi)
    start, stop = 1.2*dt, (1.2+width)*dt
    times = np.union1d(np.arange(19)*dt, [start, stop])
    contacts = np.zeros(n-1)
    contacts[n//2-1] = resistance
    links, capacity = contact_chain.cell_centered_slab_network(
        np.full(n, 1/n), 1., 1., contact_resistance=contacts, resolve_contacts=True)
    cut = n//2+1  # link between the two zero-capacity surface nodes
    x = (np.arange(n)+.5)/n
    symmetric = np.sin(pi*x)*np.sinc(.5/n)
    odd = np.where(x < .5, np.sin(k*x), -np.sin(k*(1-x)))*np.sinc(k/(2*pi*n))
    source = np.zeros_like(capacity)
    source[capacity > 0] = (symmetric + odd)/n
    states = [np.zeros_like(capacity)]
    heats, residuals = [], []
    for a, b in zip(times, times[1:]):
        power = source if start <= a < stop else np.zeros_like(source)
        end, heat = contact_chain.constant_contact_interval(
            links, capacity, b-a, states[-1], nodal_power=power,
            t_left=0., t_right=0., allow_massless=True)
        residuals.append((capacity*(end-states[-1])-(b-a)*power)[1:-1]+np.diff(heat))
        states.append(end)
        heats.append(heat[cut])
    states = np.asarray(states)
    active = np.clip(times-start, 0, stop-start)
    off = np.maximum(times-stop, 0)
    amplitude = -np.expm1(-k*k*active)*np.exp(-k*k*off)/(k*k)
    jump = states[:, cut]-states[:, cut+1]
    exact_jump = 2*math.sin(k/2)*amplitude
    # Integrate a' + k²a = 1 during heating and zero forcing afterwards.
    exact_heat = -math.cos(k/2)/k*(np.diff(active)-np.diff(amplitude))
    first = np.searchsorted(times, stop)
    left = first + pulse.reversal_bracket_index(states[first:, -2]/links[-1])

    def wall_power(t):
        if t == times[left]:
            return states[left, -2]/links[-1]
        end, _ = contact_chain.constant_contact_interval(
            links, capacity, t-times[left], states[left], t_left=0., t_right=0., allow_massless=True)
        return end[-2]/links[-1]

    root = brentq(wall_power, times[left], times[left+1], xtol=1e-14)
    wall_amplitudes = np.array([pi, k])*(-np.expm1(-rates*(stop-start)))/rates
    exact_root = stop + math.log(wall_amplitudes[1]/wall_amplitudes[0])/float(rates[1]-rates[0])
    heat_scale = abs(exact_heat).max()
    return dict(cells=n, resistance=resistance, width_over_interval=width, antisymmetric_wavenumber=k,
                exact_reversal_time=exact_root, measured_reversal_time=root,
                timing_error_over_interval=abs(root-exact_root)/dt,
                relative_jump_error=float(abs(jump-exact_jump).max()/abs(exact_jump).max()),
                relative_contact_heat_error=float(abs(heats-exact_heat).max()/heat_scale),
                max_local_balance_over_heat_scale=float(np.abs(residuals).max()/heat_scale))


if __name__ == "__main__":
    runs = []
    for resistance in (.2, 1.):
        for width in (.05, .5):
            previous = None
            for n in (16, 32, 64):
                row = measure(n, resistance, width)
                keys = ("timing_error_over_interval", "relative_jump_error", "relative_contact_heat_error")
                orders = {} if previous is None else {key: math.log2(previous[key]/row[key]) for key in keys}
                row.update(orders=orders, passed=row["max_local_balance_over_heat_scale"] < 1e-9 and
                           (n < 64 or (min(orders.values()) > 1.7 and max(row[key] for key in keys) < .01)))
                runs.append(row)
                previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(f) for f in
                          (__file__, pulse.__file__, contact_chain.__file__)],
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDEM/src python examples/dem_resistive_pulse.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
