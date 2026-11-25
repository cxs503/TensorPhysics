"""Asymmetrically coupled sheet heated by a local sinusoidal source between cold walls."""
import argparse
from dataclasses import replace
import json

import numpy as np
import torch

import interface_heat_storage as slab
from conduction_provenance import provenance
from tensordem.thermal import contact_chain
from tensorfem import thermal


def measure(n, fraction, initialization):
    omega, capacity, resistance, drive = 2*np.pi, .8, .3, 1.+.4j
    # Each homogeneous half-slab is terminated by zero temperature. Its
    # surface thermal impedance is tanh(lambda*L)/(k*lambda).
    k = np.array([1., 4.])
    wave = np.sqrt(1j*omega*np.array([1., 2.])/k)
    material_z = np.tanh(.5*wave)/(k*wave)
    admittance = 1/(material_z+resistance*np.array([fraction, 1-fraction]))
    sheet = drive/(1j*omega*capacity+admittance.sum())
    flux = admittance*sheet  # positive away from the sheet on both sides
    time = np.arange(8*n+1)/(8*n)
    dt = np.diff(time)
    phase = np.exp(1j*omega*time)
    exact_heat = (np.diff(phase)[:,None]*flux/(1j*omega)).real
    supplied = .5*dt*((drive*phase[:-1]).real+(drive*phase[1:]).real)
    r, c, model = slab.models(n, resistance, capacity, fraction)
    mid = n//2
    # Initialize on each spatial discretization's periodic orbit. Projecting
    # the continuum profile excites extra decaying modes at the low-resistance
    # contact; their startup flux is a different initial-value problem.
    generation = np.zeros_like(c, dtype=complex)
    generation[mid+1] = drive
    initial = contact_chain.harmonic_contact_temperatures(r, c, omega, t_left=0., t_right=0.,
                                                         nodal_power=generation).real
    if initialization == "continuum":
        x = np.arange(n//2+1)/n
        center = .5*(x[1:]+x[:-1])
        surface = material_z*flux
        def profile(side, distance):
            return surface[side]*np.sinh(wave[side]*distance)/np.sinh(wave[side]*.5)
        avg = np.sinh(wave/(2*n))/(wave/(2*n))
        initial = np.r_[0., avg[0]*profile(0, center), sheet, avg[1]*profile(1, center[::-1]), 0.].real
    powers = np.zeros((len(time), len(c)))
    powers[:,mid+1] = (drive*phase).real
    t = contact_chain.transient_contact_temperatures(r, c, time, initial, t_left=0., t_right=0.,
                                                   nodal_power=powers)
    links = -np.diff(t, axis=1)[:,mid:mid+2]/r[mid:mid+2]*[-1,1]
    dem = (t[:,mid+1], .5*dt[:, None]*(links[:-1]+links[1:]))

    model = replace(model, fixed_temperatures=torch.zeros_like(model.fixed_temperatures))
    generation = torch.zeros(len(model.nodes), dtype=torch.complex128)
    generation[-1] = drive
    initial = thermal.solve_harmonic_thermal(replace(model, nodal_heat=generation), omega,
                                             sparse=True).temperature.real
    if initialization == "continuum":
        initial = torch.from_numpy(np.r_[profile(0,x), profile(1,x[::-1]), sheet].real)
    powers = torch.zeros((len(time), len(model.nodes)), dtype=model.nodes.dtype)
    powers[:, -1] = torch.from_numpy((drive*phase).real)
    result = thermal.solve_transient_thermal(model, torch.from_numpy(time), initial,
                                             .5, sparse=True, nodal_heat_history=powers)
    energy = thermal.thermal_element_nodal_heat_steps(model, result).numpy()
    fem = (result.temperature[:, -1].numpy(), np.column_stack((energy[:,mid-1,1], energy[:,mid,0])))

    for name, (temperature, heat) in (("DEM", dem), ("FEM", fem)):
        residual = capacity*np.diff(temperature)+heat.sum(axis=1)-supplied
        errors = np.r_[abs(temperature-(sheet*phase).real).max(),
                       abs((heat-exact_heat)/dt[:,None]).max(axis=0)]
        yield name, errors, float(abs(residual).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", choices=("periodic", "continuum"), default="periodic")
    initialization = parser.parse_args().initialization
    rows = []
    for fraction in (.15, .85):
        previous = {}
        for n in (8, 16, 32, 64, 128):
            for name, errors, balance in measure(n, fraction, initialization):
                orders = np.log2(previous[name]/errors) if name in previous else None
                passed = np.isfinite(errors).all() and balance < 1e-12
                if orders is not None:
                    passed &= min(orders) > (1.9 if n == 128 else 0.)
                rows.append(dict(solver=name, cells=n, steps=8*n, left_resistance_fraction=fraction,
                                 sheet_temperature_error_K=errors[0], left_mean_flux_error_W_m2=errors[1],
                                 right_mean_flux_error_W_m2=errors[2], max_step_balance_J_m2=balance,
                                 orders=orders.tolist() if orders is not None else None, passed=bool(passed)))
                previous[name] = errors
    print(json.dumps(dict(initialization=initialization, runs=rows, sources=[provenance(f) for f in
                    (__file__, slab.__file__, thermal.__file__, contact_chain.__file__)]), indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
