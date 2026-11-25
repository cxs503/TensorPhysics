"""Harmonic response to out-of-phase bulk heating and ambient temperature."""
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, omega, ambient):
    k, capacity, loss = 1.7, 1.3, 4.
    # Each load has the first sine profile on a unit slab with zero wall phasors.
    # Projecting the continuum equation gives this scalar complex response.
    heating = .8-.25j
    amplitude = (heating + loss*ambient)/(k*math.pi**2 + loss + 1j*omega*capacity)
    cells = (torch.arange(n, dtype=torch.float64)+.5)/n
    faces = torch.arange(n+1, dtype=torch.float64)/n
    shape = torch.sin(math.pi*cells)[None, :]
    field, flux = heat.solve_harmonic_slab_2d(
        heating*shape, 1/n, (0., 0.), omega, conductivity=k, volumetric_heat_capacity=capacity,
        heat_loss_coefficient=loss, ambient_temperature=ambient*shape, return_x_flux=True)
    wall_scale = k*math.pi*abs(amplitude)
    exact_flux = -k*math.pi*amplitude*torch.cos(math.pi*faces)
    return dict(columns=n, omega_rad_s=omega, ambient_phasor_K=[ambient.real, ambient.imag],
                max_temperature_error_over_amplitude=float((field-amplitude*shape).abs().max()/abs(amplitude)),
                max_flux_error_over_wall_amplitude=float((flux-exact_flux).abs().max()/wall_scale),
                max_wall_phase_error_rad=float(torch.angle(flux[[0,-1]]/exact_flux[[0,-1]]).abs().max()))


if __name__ == "__main__":
    runs = []
    for ambient in (.15+.4j, -.3-.1j):
        for omega in (.5, 20., 120.):
            previous = None
            for n in (16, 32, 64, 128):
                row = measure(n, omega, ambient)
                row["passed"] = True
                for metric in ("max_temperature_error_over_amplitude", "max_flux_error_over_wall_amplitude"):
                    if previous:
                        order = math.log2(previous[metric]/row[metric])
                        row[metric + "_order"] = order
                        row["passed"] &= order > 1.8
                    if n == 128:
                        row["passed"] &= row[metric] < .001
                if n == 128:
                    row["passed"] &= row["max_wall_phase_error_rad"] < .001
                runs.append(row)
                previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_harmonic_ambient_exchange.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
