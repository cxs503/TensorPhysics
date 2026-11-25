"""Periodic response of two storing rings with quasi-steady conduction in three annular materials."""
import argparse
import json
import math

import numpy as np

from conduction_provenance import provenance


RADII = (.4, .55, .75, .9, 1.1)  # bore, sheet, artificial join, sheet, exterior
K = (1., 2.5, .8)
CONTACT = (.5, .9)
CAPACITY = (.25, .8)
WALLS = np.array([1., .3j])


def continuum(omega, walls=WALLS, generation=(0., 0.), fractions=(.5, .5)):
    # Exact radial resistances in K m/W, with heat rates per unit axial length.
    a, r1, _, r2, b = RADII
    surface = 2*np.pi*np.array([r1, r2])
    contact = np.asarray(CONTACT)/surface
    inner = contact*fractions
    outer = contact-inner
    resistance = np.log([r1/a, r2/r1, b/r2])/(2*np.pi*np.array(K))
    resistance += [inner[0], outer[0]+inner[1], outer[1]]
    g0, g1, g2 = 1/resistance
    matrix = np.array([[g0+g1, -g1], [-g1, g1+g2]]) + 1j*omega*np.diag(surface*CAPACITY)
    temp = np.linalg.solve(matrix, walls*[g0, g2] + surface*generation)
    power = -np.diff(np.r_[walls[0], temp, walls[1]])/resistance
    return np.column_stack((temp, power[:-1]/surface, power[1:]/surface))


def angular_reference(omega, mode, walls=WALLS, generation=(0., 0.), fractions=(.5, .5), sheet_k=(0., 0.)):
    # In a source-free annulus T_m = A*r^m+B*r^-m. Propagate the pair
    # (T_m, r*q_m) between the two physical walls without an artificial join.
    edges = [RADII[0], RADII[1], RADII[3], RADII[4]]
    def propagate(initial):
        t, p = walls[0], initial
        sheets = []
        for i, k in enumerate(K):
            radius = edges[i+1]
            z = mode*math.log(radius/edges[i])
            c, s = math.cosh(z), math.sinh(z)
            t, p = c*t-s*p/(k*mode), c*p-s*k*mode*t
            if i < 2:
                incoming = p/radius
                t -= CONTACT[i]*fractions[i]*incoming
                outgoing = incoming+generation[i]-(1j*omega*CAPACITY[i]+sheet_k[i]*(mode/radius)**2)*t
                sheets.append([t, incoming, outgoing])
                p = radius*outgoing
                t -= CONTACT[i]*(1-fractions[i])*outgoing
        return t, np.asarray(sheets)
    offset, _ = propagate(0.)
    unit, _ = propagate(1.)
    end, fields = propagate((walls[1]-offset)/(unit-offset))
    if abs(end-walls[1]) > 1e-10:
        raise ArithmeticError("angular reference missed exterior temperature")
    return fields


if __name__ == "__main__":
    from tensorbem import harmonic_rings, layered_annulus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--sheet-driven", action="store_true", help="drive the two sheets in quadrature with isothermal walls")
    parser.add_argument("--inner-fractions", nargs=2, type=float, default=(.5, .5),
                        metavar=("RING1", "RING2"), help="fraction of each contact resistance on its inner side")
    parser.add_argument("--sheet-conductivities", nargs=2, type=float, default=(0., 0.),
                        metavar=("RING1", "RING2"), help="circumferential conductivity times thickness in W/K")
    args = parser.parse_args()
    sheet_k = np.array(args.sheet_conductivities)
    fractions = args.inner_fractions
    mode = args.mode
    walls = np.zeros(2) if args.sheet_driven else WALLS
    generation = np.array([1., 0.6j]) if args.sheet_driven else np.zeros(2)
    runs = []
    for formulation in ("circular", "polygonal"):
        previous = {}
        for n in ((16, 32, 64, 128) if mode else (16, 32, 64)):
            response = harmonic_rings.assemble_ring_response(
                (RADII[0], RADII[1], RADII[3], RADII[4]), K, CONTACT, n, mode=mode,
                formulation=formulation, join_radius=RADII[2], inner_resistance_fractions=fractions,
                sheet_conductivities=sheet_k)
            # For straight panels the discrete angular gradient uses the chord length.
            angular_factor = mode if formulation == "circular" else np.sin(mode*np.pi/n)/np.sin(np.pi/n)
            modal_loss = sheet_k*(angular_factor/np.array([RADII[1], RADII[3]]))**2
            leakage = response.mode_leakage
            for omega in (1., 8., 24.):
                solution = response.solve(omega, walls, CAPACITY, sheet_generation=generation)
                fields = np.column_stack((solution.sheet_temperature, solution.incoming_flux, solution.outgoing_flux))
                imbalance = solution.join_power_residual
                expected = (angular_reference(omega, mode, walls, generation, fractions, sheet_k) if mode
                            else continuum(omega, walls, generation, fractions))
                ratio = fields/expected
                error = float(abs(ratio-1).max())
                order = None if omega not in previous or formulation == "circular" else math.log2(previous[omega]/error)
                sheet_balance = float(abs(generation+fields[:, 1]-fields[:, 2]-(modal_loss+1j*omega*np.asarray(CAPACITY))*fields[:, 0]).max())
                passed = leakage < 1e-9 and imbalance < 1e-10 and sheet_balance < 1e-10 and (error < 1e-9 if formulation == "circular"
                                                else n < (128 if mode else 64) or (order > 1.7 and error < .01))
                runs.append(dict(formulation=formulation, panels=n, omega=omega, max_relative_phasor_error=error,
                                 order=order, amplitude_ratios=abs(ratio).tolist(), phase_errors_deg=np.angle(ratio,deg=True).tolist(),
                                 reference_amplitudes=abs(expected).tolist(), reference_phases_deg=np.angle(expected,deg=True).tolist(),
                                 unit_response_mode_leakage=leakage, join_power_residual_W_m=imbalance, sheet_balance_residual_W_m2=sheet_balance, passed=passed))
                previous[omega] = error
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, angular_cosine_mode=mode, inner_resistance_fractions=fractions,
                          sheet_conductivities_W_K=sheet_k.tolist(), wall_temperatures_K=[[z.real, z.imag] for z in walls],
                          sheet_generation_W_m2=[[z.real, z.imag] for z in generation], model="zero bulk capacity, positive areal storage at each ring",
                          columns=["sheet temperature", "incoming radial flux", "outgoing radial flux"], runs=runs,
                          sources=[provenance(p) for p in (__file__, harmonic_rings.__file__, layered_annulus.__file__)],
                          reproduction=f"OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src python examples/bem_harmonic_storing_rings.py --mode {mode}" + (" --sheet-driven" if args.sheet_driven else "")
                          + f" --inner-fractions {fractions[0]} {fractions[1]}"
                          + f" --sheet-conductivities {sheet_k[0]} {sheet_k[1]}"),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
