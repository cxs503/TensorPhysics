"""Verify independent storage on both traces of two resistive thermal contacts."""
import argparse
import json

import numpy as np
from scipy.linalg import expm

from conduction_provenance import provenance
import harmonic_two_contact_updates as stack
from tensordg import heat_1d


def measure(n, omega, resistance, films=False):
    # All layer admittances are scaled to total power. Capacities here are J/K.
    cl, cr = np.array([.28, .51]), np.array([.65, .19])
    sl, sr = np.array([.32+.14j, -.24j]), np.array([-.15j, .17-.08j])
    # Film conductances in W/K act on the respective trace temperatures.
    hl = np.array([.45, 1.1]) if films else np.zeros(2)
    hr = np.array([.85, .32]) if films else np.zeros(2)
    al, ar = np.array([.3+.7j, -.2+.1j]), np.array([-.4+.2j, .6-.5j])
    temperature, power = .75-.24j, .32+.51j
    exact_t, exact_p, contacts = [], [], []
    for i, (area, length, k, c) in enumerate(zip(stack.AREA, stack.LENGTH, stack.K, stack.C)):
        transfer = expm(length*np.array([[0., -1/(k*area)], [-1j*omega*c*area, 0.]]))
        east_t, east_p = transfer @ [temperature, power]
        exact_t.append([temperature, east_t])
        exact_p.append([power, -east_p])
        if i < len(resistance):
            through = east_p+sl[i]+hl[i]*(al[i]-east_t)-1j*omega*cl[i]*east_t
            temperature = east_t-resistance[i]*through
            power = through+sr[i]+hr[i]*(ar[i]-temperature)-1j*omega*cr[i]*temperature
            contacts.append(through)

    drive = (exact_t[0][0], exact_t[-1][1])
    _, t, flux = heat_1d.compose_heat_wall_admittances_1d(
        stack.dg_layers(n, omega), resistance, wall_temperatures=drive,
        interface_areal_heat_capacity=cr, interface_left_areal_heat_capacity=cl,
        interface_left_film_coefficients=hl, interface_left_ambient_temperatures=al,
        interface_film_coefficients=hr, interface_ambient_temperatures=ar,
        angular_frequency=omega, interface_heat_sources=sr, interface_left_heat_sources=sl)
    t, flux = t.numpy(), flux.numpy()
    exact_t = np.asarray(exact_t)
    left_rate = 1j*omega*cl*t[:-1, 1]
    right_rate = 1j*omega*cr*t[1:, 0]
    left_input = hl*(al-t[:-1, 1])
    right_input = hr*(ar-t[1:, 0])
    through = sl+left_input-flux[:-1, 1]-left_rate
    balance = through+sr+right_input-right_rate-flux[1:, 0]
    exchange = {}
    if films:
        exchange["left_film_error_W"] = float(abs(left_input-hl*(al-exact_t[:-1, 1])).max())
        exchange["right_film_error_W"] = float(abs(right_input-hr*(ar-exact_t[1:, 0])).max())
    return dict(**exchange, cells_per_layer=n, omega_rad_s=omega, resistance_K_W=resistance.tolist(),
                temperature_error_K=float(abs(t-exact_t).max()),
                material_power_error_W=float(abs(flux-exact_p).max()),
                contact_power_error_W=float(abs(through-contacts).max()),
                left_storage_error_W=float(abs(left_rate-1j*omega*cl*exact_t[:-1, 1]).max()),
                right_storage_error_W=float(abs(right_rate-1j*omega*cr*exact_t[1:, 0]).max()),
                jump_residual_K=float(abs(t[:-1, 1]-t[1:, 0]-resistance*through).max()),
                contact_balance_W=float(abs(balance).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--films", action="store_true", help="apply independently driven films to both contact traces")
    films = parser.parse_args().films
    rows = []
    metrics = ["temperature_error_K", "material_power_error_W", "contact_power_error_W",
               "left_storage_error_W", "right_storage_error_W"]
    if films:
        metrics += ["left_film_error_W", "right_film_error_W"]
    for omega in (.5, 2., 4.):
        for resistance in (stack.BASE, np.array([0., .18])):
            previous = None
            for n in (2, 4, 8, 16):
                row = measure(n, omega, resistance, films)
                errors = np.array([row[k] for k in metrics])
                orders = np.log2(previous/errors).tolist() if previous is not None else []
                row.update(orders=orders, passed=bool(all(p > 3.5 for p in orders)
                           and (n < 16 or max(errors) < 2e-6)
                           and max(row["contact_balance_W"], row["jump_residual_K"]) < 1e-10))
                rows.append(row)
                previous = errors
    print(json.dumps(dict(reference="Exact three-layer transfer matrices with separate left/right storage and applied phasor powers",
                          reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorDG python examples/dg_harmonic_contact_storage.py"+(" --films" if films else ""),
                          order_metrics=metrics, sources=[provenance(f) for f in (__file__, stack.__file__, heat_1d.__file__)],
                          runs=rows), allow_nan=False, indent=2))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
