"""Compare an ambient-driven material interface to its exact harmonic heat flux."""
import argparse
import cmath
import json
import math

import torch
from tensorsph import heat_conduction_2d as heat

from conduction_provenance import provenance


def fem_contact(conductivity, capacity, loss, ambient, omega, resistance, walls):
    from tensorfem import thermal

    n = len(conductivity)
    mid = n//2
    x = torch.linspace(0., 1., n+1, dtype=torch.float64)
    start = torch.arange(n)
    contact = {}
    if resistance:
        x = torch.cat((x[:mid+1], x[mid:]))
        start[mid:] += 1
        contact = dict(contact_pairs=torch.tensor([[mid, mid+1]]), contact_conductance=1/resistance)
    fixed = [node for node, wall in zip((0, len(x)-1), walls) if wall is not None]
    prescribed = [wall for wall in walls if wall is not None]
    model = thermal.ThermalModel(
        x[:, None], torch.stack((start, start+1), dim=1), conductivity, 1., capacity,
        torch.tensor(fixed, dtype=torch.long), torch.tensor(prescribed, dtype=torch.complex128),
        heat_source=loss*ambient, heat_loss_coefficient=loss, **contact)
    if omega == 0:
        state = thermal.solve_low_frequency_thermal(model).steady
    else:
        state = thermal.solve_harmonic_thermal(model, omega)
    field = state.temperature
    inward = thermal.thermal_element_nodal_power(model, field, omega)
    left_flux = -complex(inward[mid-1, 1])
    right_flux = complex(inward[mid, 0])
    return complex(field[start[mid-1]+1]), complex(field[start[mid]]), left_flux, right_flux


def measure(n, omega, resistance, walls, backend, compare_fvm=False, compare_fem=False):
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
    comparison = {}
    if compare_fvm:
        import numpy as np
        from tensorfvm.cht import rod
        # Unit cross-section converts the rod's power in W to the slab flux in W/m².
        _, _, entering, leaving, traces = rod.solve_harmonic_stepped_rod(
            np.full(n, 1/n), np.ones(n), conductivity.numpy(), layers(capacity).numpy(),
            tuple(0. if w is None else w for w in walls), omega,
            end_types=tuple("power" if w is None else "temperature" for w in walls),
            heat_loss_coefficient=layers(loss).numpy(), bulk_ambient_temperature=layers(ambient, torch.complex128).numpy(),
            contact_resistance=contact.numpy(), return_traces=True)
        q = leaving[i]
        tl, tr = traces[0][i], traces[1][i]
        comparison = dict(fvm_flux_relative_error=float(abs(q/exact_flux-1)),
                          fvm_max_trace_error_K=float(max(abs(tl-exact_left), abs(tr-exact_right))),
                          fvm_interface_imbalance_W_m2=float(abs(q-entering[i+1])),
                          solver_flux_difference_W_m2=float(abs(q-flux)),
                          solver_trace_difference_K=float(max(abs(tl-complex(left[i])), abs(tr-complex(right[i])))))
    if compare_fem:
        fl, fr, ql, qr = fem_contact(conductivity, layers(capacity), layers(loss),
                                     layers(ambient, torch.complex128), omega, resistance, walls)
        comparison.update(fem_flux_relative_error=abs(ql/exact_flux-1),
                          fem_max_trace_error_K=max(abs(fl-exact_left), abs(fr-exact_right)),
                          fem_interface_imbalance_W_m2=abs(ql-qr),
                          fem_jump_error_K=abs(fl-fr-resistance*exact_flux),
                          fem_sph_flux_difference_W_m2=abs(ql-flux))
        if compare_fvm:
            comparison["fem_fvm_flux_difference_W_m2"] = float(abs(ql-q))
    return dict(**comparison, columns=n, omega_rad_s=omega, contact_resistance_m2_K_W=resistance, backend=backend,
                wall_conditions=["insulated" if w is None else "zero_temperature_phasor" for w in walls],
                exact_flux_W_m2=[exact_flux.real, exact_flux.imag], flux_relative_error=abs(flux/exact_flux-1),
                flux_phase_error_rad=abs(cmath.phase(flux/exact_flux)),
                max_trace_error_K=max(abs(complex(left[i])-exact_left), abs(complex(right[i])-exact_right)),
                jump_error_K=abs(complex(left[i]-right[i])-resistance*exact_flux))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-fvm", action="store_true", help="also verify the unit-area finite-volume rod")
    parser.add_argument("--compare-fem", action="store_true", help="also verify the unit-area linear finite elements")
    args = parser.parse_args()
    files = [__file__, heat.__file__]
    path = "../TensorSPH/src"
    metrics = ["flux_relative_error", "max_trace_error_K"]
    if args.compare_fvm:
        from tensorfvm.cht import rod
        files.append(rod.__file__)
        path += ":../TensorFVM/src"
        metrics.extend(["fvm_flux_relative_error", "fvm_max_trace_error_K"])
    if args.compare_fem:
        from tensorfem import thermal
        files.append(thermal.__file__)
        path += ":../TensorFEM/src"
        metrics.extend(["fem_flux_relative_error", "fem_max_trace_error_K"])
    runs = []
    for backend in ("torch", "banded"):
        for walls in ((0., 0.), (None, 0.), (None, None)):
            for omega in (0., 12.):
                for resistance in (0., .25):
                    previous = None
                    for n in (16, 32, 64):
                        row = measure(n, omega, resistance, walls, backend, args.compare_fvm, args.compare_fem)
                        row["passed"] = True
                        for metric in metrics:
                            if previous:
                                order = math.log2(previous[metric]/row[metric])
                                row[metric + "_order"] = order
                                row["passed"] &= order > 1.7
                        if n == 64:
                            row["passed"] &= (row["flux_relative_error"] < .005 and row["max_trace_error_K"] < .001
                                              and row["jump_error_K"] < .001 and row["flux_phase_error_rad"] < .005)
                        if args.compare_fvm:
                            row["passed"] &= row["fvm_interface_imbalance_W_m2"] < 1e-10
                            if n == 64:
                                row["passed"] &= row["fvm_flux_relative_error"] < .005 and row["fvm_max_trace_error_K"] < .001
                        if args.compare_fem:
                            row["passed"] &= row["fem_interface_imbalance_W_m2"] < 1e-10
                            if n == 64:
                                row["passed"] &= (row["fem_flux_relative_error"] < .005 and row["fem_max_trace_error_K"] < .001
                                                  and row["fem_jump_error_K"] < .001)
                        runs.append(row)
                        previous = row
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(f) for f in files],
                          reproduction=f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH={path} python examples/sph_ambient_driven_contact.py"
                          + (" --compare-fvm" if args.compare_fvm else "") + (" --compare-fem" if args.compare_fem else "")),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
