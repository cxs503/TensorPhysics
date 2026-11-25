"""Separate contact-chain spatial and temporal flux errors over one fixed observation window."""
import json
from inspect import getfile

import numpy as np

from conduction_provenance import provenance
from interface_heat_generation import projected_profiles, sheet_reference
from tensordem.thermal import contact_chain


def diagnose(n, fraction, duration, substeps):
    omega, cv_sheet, resistance, drive, wave, sheet, flux, surface = sheet_reference(fraction)
    mid = n//2
    left, right, _, _ = projected_profiles(n, wave, surface)
    initial = np.r_[0., left, sheet, right, 0.].real
    contacts = np.zeros(n-1)
    sheets = contacts.copy()
    contacts[mid-1], sheets[mid-1] = resistance, cv_sheet
    r, c = contact_chain.cell_centered_slab_network(np.full(n, 1/n), np.repeat([1.,4.], mid),
                     np.repeat([1.,2.], mid), contact_resistance=contacts, interface_heat_capacity=sheets,
                     left_resistance_fraction=fraction)
    source = np.zeros(len(c), dtype=complex)
    source[mid+1] = drive
    end, heat = contact_chain.harmonic_contact_interval(r, c, omega, 0., duration, initial,
                                      t_left=0., t_right=0., nodal_power=source)
    integral_phase = np.expm1(1j*omega*duration)/(1j*omega)
    outward = heat[mid:mid+2]*[-1, 1]
    balance = abs(cv_sheet*(end[mid+1]-initial[mid+1])+outward.sum()-(drive*integral_phase).real)
    spatial = (outward-(flux*integral_phase).real)/duration
    time = np.linspace(0., duration, substeps+1)
    powers = (np.exp(1j*omega*time[:, None])*source).real
    t = contact_chain.transient_contact_temperatures(r, c, time, initial, t_left=0., t_right=0., nodal_power=powers)
    contact = -np.diff(t, axis=1)[:,mid:mid+2]/r[mid:mid+2]*[-1,1]
    computed = np.diff(time) @ (.5*(contact[1:]+contact[:-1]))
    return spatial, (computed-outward)/duration, balance


def main():
    duration = 1/1024
    rows = []
    for fraction in (.15, .85):
        for study, ladder in (("space", (32,64,128,256,512)), ("time", (4,8,16,32,64,128))):
            previous = None
            for level in ladder:
                n, steps = (level, 1) if study == "space" else (128, level)
                spatial, temporal, balance = diagnose(n, fraction, duration, steps)
                errors = abs(spatial if study == "space" else temporal)
                orders = np.log2(previous/errors) if previous is not None else None
                passed = np.isfinite(errors).all() and balance < 1e-12
                if level == ladder[-1]:
                    passed &= min(orders) > 1.9
                rows.append(dict(study=study, left_resistance_fraction=fraction, cells=n, substeps=steps,
                                 mean_flux_errors_W_m2=errors.tolist(), balance_J_m2=balance,
                                 orders=orders.tolist() if orders is not None else None, passed=bool(passed)))
                previous = errors
    # In the original coupled ladder, the first output interval shrinks with h.
    # Retain signs so cancellation between the two error sources is observable.
    coupled = []
    for fraction in (.15, .85):
        for n in (32, 64, 128, 256):
            window = 1/(8*n)
            space, time, _ = diagnose(n, fraction, window, 1)
            coupled.append(dict(left_resistance_fraction=fraction, cells=n, first_interval_s=window,
                                spatial_bias_W_m2=space.tolist(), temporal_bias_W_m2=time.tolist(),
                                combined_bias_W_m2=(space+time).tolist()))
    print(json.dumps(dict(window_s=duration, coupled_first_interval=coupled, runs=rows, sources=[provenance(f) for f in
                   (__file__, getfile(sheet_reference), contact_chain.__file__)]), indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
