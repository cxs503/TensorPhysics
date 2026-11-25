"""Compare an ambient-driven material interface to its exact harmonic heat flux."""
import cmath
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def measure(n, omega, resistance, walls, backend):
    k = (1., 2.5)
    capacity = (1.3, .7)
    loss = (2., 4.)
    ambient = (.8-.1j, -.2+.55j)
    # Solve k*T''=(H+i*w*C)*T-H*Ta on each half slab. At the interface
    # the left and right traces are free[0]-Z[0]*q and free[1]+Z[1]*q.
    free, impedance = [], []
    for conductivity, cv, h, ta, wall in zip(k, capacity, loss, ambient, walls):
        shift = h + 1j*omega*cv
        wave = cmath.sqrt(shift/conductivity)
        particular = h*ta/shift
        z = cmath.tanh(.5*wave)
        if wall is None:
            free.append(particular)
            impedance.append(1/(conductivity*wave*z))
        else:
            free.append(particular+(wall-particular)/cmath.cosh(.5*wave))
            impedance.append(z/(conductivity*wave))
    exact_flux = (free[0]-free[1])/(resistance+sum(impedance))
    exact_left = free[0]-impedance[0]*exact_flux
    exact_right = free[1]+impedance[1]*exact_flux

    def layers(values, dtype=torch.float64):
        return torch.tensor(values, dtype=dtype).repeat_interleave(n//2)

    contact = torch.zeros(n-1, dtype=torch.float64)
    i = n//2-1
    contact[i] = resistance
    conductivity = layers(k)
    field, cuts = heat.solve_harmonic_layered_slab(
        torch.zeros(n), 1/n, walls, omega, conductivity=conductivity,
        volumetric_heat_capacity=layers(capacity), contact_resistance_x=contact,
        heat_loss_coefficient=layers(loss), ambient_temperature=layers(ambient, torch.complex128),
        return_x_flux=True, backend=backend)
    left, right = heat.layered_slab_contact_temperatures(field, cuts, 1/n, conductivity)
    flux = complex(cuts[i+1])
    return dict(columns=n, omega_rad_s=omega, contact_resistance_m2_K_W=resistance, backend=backend,
                wall_conditions=["insulated" if w is None else "zero_temperature_phasor" for w in walls],
                exact_flux_W_m2=[exact_flux.real, exact_flux.imag], flux_relative_error=abs(flux/exact_flux-1),
                flux_phase_error_rad=abs(cmath.phase(flux/exact_flux)),
                max_trace_error_K=max(abs(complex(left[i])-exact_left), abs(complex(right[i])-exact_right)),
                jump_error_K=abs(complex(left[i]-right[i])-resistance*exact_flux))


if __name__ == "__main__":
    runs = []
    for backend in ("torch", "banded"):
        for walls in ((0., 0.), (None, 0.), (None, None)):
            for omega in (0., 12.):
                for resistance in (0., .25):
                    previous = None
                    for n in (16, 32, 64):
                        row = measure(n, omega, resistance, walls, backend)
                        row["passed"] = True
                        for metric in ("flux_relative_error", "max_trace_error_K"):
                            if previous:
                                order = math.log2(previous[metric]/row[metric])
                                row[metric + "_order"] = order
                                row["passed"] &= order > 1.7
                        if n == 64:
                            row["passed"] &= (row["flux_relative_error"] < .005 and row["max_trace_error_K"] < .001
                                              and row["jump_error_K"] < .001 and row["flux_phase_error_rad"] < .005)
                        runs.append(row)
                        previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(__file__), provenance(heat.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=../TensorSPH/src python examples/sph_ambient_driven_contact.py"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
