"""Compare BEM and FVM at fixed physical loads in a three-material quasi-steady annulus."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import harmonic_rings, layered_annulus
from tensorfvm.cht import annulus, storing_rings

EDGES = np.array([.4, .55, .9, 1.1])
K = np.array([1., 2.5, .8])
CONTACT = np.array([.5, .9])
FRACTIONS = np.array([.2, .7])
CAPACITY = np.array([.25, .8])
WALLS = np.array([.7-.2j, -.3+.5j])
VOLUME = np.array([1.+.2j, -.4+.7j, .3-.5j])
SHEETS = np.array([.2j, -.3+.4j])


RESERVOIRS = np.array([-.5+.3j, .4+.8j])


def continuum(mode, omega, films=(None, None), fractions=FRACTIONS, sheet_h=(0., 0.)):
    # Six integration constants, two per layer. Last column is the particular
    # solution of the Poisson equation with source V_i*(r/R)^m.
    def traces(layer, r):
        temp, flux = np.zeros((2, 7), dtype=complex)
        pair = slice(2*layer, 2*layer+2)
        if mode:
            temp[pair] = [r**mode, r**(-mode)]
            flux[pair] = -K[layer]*mode/r*np.array([r**mode, -r**(-mode)])
        else:
            temp[pair] = [1., np.log(r)]
            flux[pair] = [0., -K[layer]/r]
        temp[-1] = -VOLUME[layer]*r**(mode+2)/(4*(mode+1)*K[layer]*EDGES[-1]**mode)
        flux[-1] = -K[layer]*(mode+2)*temp[-1]/r
        return temp, flux

    bore, q_bore = traces(0, EDGES[0])
    exterior, q_ext = traces(2, EDGES[-1])
    equations = [bore if films[0] is None else bore+q_bore/films[0],
                 exterior if films[1] is None else exterior-q_ext/films[1]]
    loads = list(WALLS)
    samples = []
    material_traces = []
    for i, r in enumerate(EDGES[1:-1]):
        left, qi = traces(i, r)
        right, qo = traces(i+1, r)
        sheet = left - CONTACT[i]*fractions[i]*qi
        equations.extend((sheet-right-CONTACT[i]*(1-fractions[i])*qo,
                          qo-qi+(sheet_h[i]+1j*omega*CAPACITY[i])*sheet))
        loads.extend((0., SHEETS[i]+sheet_h[i]*RESERVOIRS[i]))
        samples.append((sheet, qi, qo))
        material_traces.extend((left, right))
    matrix = np.asarray(equations)
    coeff = np.r_[np.linalg.solve(matrix[:, :-1], loads-matrix[:, -1]), 1.]
    t, qi, qo = (np.array(samples) @ coeff).T
    rates = 2*np.pi*EDGES[[0,-1]]*([q_bore, q_ext] @ coeff)
    return np.r_[t, qi, qo, rates, bore @ coeff, exterior @ coeff, np.array(material_traces) @ coeff]


def measure(mode, films=(None, None), fractions=FRACTIONS, sheet_h=(0., 0.)):
    rows = []
    previous = {}
    for n in (4, 8, 16, 32):
        mesh = np.r_[np.concatenate([np.geomspace(a,b,n+1)[:-1] for a,b in zip(EDGES, EDGES[1:])]), EDGES[-1]]
        lo, hi = mesh[:-1], mesh[1:]
        # The FVM load is the exact shell-volume mean of the BEM source density.
        source = (np.repeat(VOLUME, n)*2*(hi**(mode+2)-lo**(mode+2))
                  / ((mode+2)*EDGES[-1]**mode*(hi**2-lo**2)))
        bem = harmonic_rings.assemble_ring_response(EDGES, K, CONTACT, 4*n, mode=mode,
            inner_resistance_fractions=fractions, include_volume_sources=True, formulation="polygonal", film_coefficients=films)
        for omega in (1., 8., 24.):
            reference = continuum(mode, omega, films, fractions, sheet_h)
            sol = bem.solve(omega, WALLS, CAPACITY, sheet_generation=SHEETS, volume_generation=VOLUME,
                            sheet_film_coefficients=sheet_h, sheet_ambient_temperatures=RESERVOIRS)
            boundary = np.r_[sol.sheet_temperature, sol.incoming_flux, sol.outgoing_flux, sol.wall_heat_rates, sol.wall_surface_temperatures, sol.contact_temperatures.ravel()]
            fv = storing_rings.solve_storing_rings(mesh, np.repeat(K,n), [n, 2*n], WALLS, omega,
                contact_resistances=CONTACT, sheet_capacities=CAPACITY, sheet_generation=SHEETS,
                inner_resistance_fractions=fractions, azimuthal_mode=mode, source=source, return_wall_rates=True,
                sheet_film_coefficients=sheet_h, sheet_ambient_temperatures=RESERVOIRS,
                film_coefficients=films, return_wall_temperatures=True, return_contact_temperatures=True)
            volume = np.concatenate([part.ravel() for part in fv])
            errors = np.array([max(abs((x-reference)/reference)) for x in (boundary, volume)])
            difference = abs((boundary-volume)/reference)
            orders = np.log2(previous[omega]/errors) if omega in previous else None
            # Axisymmetric conduction with constant layer source uses exact half-shell drops.
            fv_pass = errors[1] < 1e-10 if not mode else n < 32 or (errors[1] < .001 and orders[1] > 1.8)
            passed = fv_pass and (n < 32 or (errors[0] < .01 and orders[0] > 1.7 and max(difference) < .01))
            reference_jump = reference[-4::2]-reference[-3::2]
            jump_errors = [max(abs((x[-4::2]-x[-3::2])/reference_jump-1)) for x in (boundary, volume)]
            passed &= n < 32 or (jump_errors[0] < .01 and jump_errors[1] < .001)
            balances = []
            for field in (boundary, volume):
                sheet, incoming, outgoing = field[:2], field[2:4], field[4:6]
                net = SHEETS + sheet_h*(RESERVOIRS-sheet) - 1j*omega*CAPACITY*sheet
                balances.append(float(max(abs(net + incoming - outgoing))))
            passed &= max(balances) < 1e-10
            rows.append(dict(bem_sheet_balance_W_m2=balances[0], fvm_sheet_balance_W_m2=balances[1], bem_jump_relative_error=jump_errors[0], fvm_jump_relative_error=jump_errors[1], mode=mode, omega_rad_s=omega, bem_panels=4*n, fvm_cells_per_layer=n,
                bem_relative_error=errors[0], fvm_relative_error=errors[1],
                bem_order=None if orders is None else orders[0],
                fvm_order=None if orders is None or not mode else orders[1],
                relative_solver_difference=difference.tolist(),
                reference_phasors=[[z.real, z.imag] for z in reference],
                bem_phasors=[[z.real, z.imag] for z in boundary],
                fvm_phasors=[[z.real, z.imag] for z in volume], passed=bool(passed)))
            previous[omega] = errors
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film-walls", choices=("none", "bore", "exterior", "both"), default="none")
    parser.add_argument("--inner-fractions", type=float, nargs=2, default=FRACTIONS, metavar=("RING1", "RING2"))
    parser.add_argument("--sheet-films", nargs=2, type=float, default=(0., 0.), metavar=("RING1", "RING2"))
    args = parser.parse_args()
    case = args.film_walls
    fractions = args.inner_fractions
    films = (7. if case in ("bore", "both") else None, 3. if case in ("exterior", "both") else None)
    runs = [row for mode in (0, 1, 2) for row in measure(mode, films, fractions, args.sheet_films)]
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, film_coefficients_W_m2_K=films,
        sheet_film_coefficients_W_m2_K=args.sheet_films,
        sheet_ambient_K=[[z.real,z.imag] for z in RESERVOIRS],
        inner_resistance_fractions=list(fractions), fixed_boundary_data_K=[[z.real, z.imag] for z in WALLS],
        phasor_encoding="[real, imaginary], exp(i omega t)",
        columns=["sheet T1 (K)", "sheet T2 (K)", "incoming q1 (W/m²)", "incoming q2 (W/m²)",
                 "outgoing q1 (W/m²)", "outgoing q2 (W/m²)", "bore rate (W/m)", "exterior rate (W/m)",
                 "bore surface temperature (K)", "exterior surface temperature (K)",
                 "ring 1 inner material (K)", "ring 1 outer material (K)",
                 "ring 2 inner material (K)", "ring 2 outer material (K)"],
        sources=[provenance(p) for p in (__file__, harmonic_rings.__file__, layered_annulus.__file__,
                                         storing_rings.__file__, annulus.__file__)],
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src:../TensorFVM/src python examples/cross_volume_heated_rings.py" + f" --film-walls {case} --inner-fractions {fractions[0]} {fractions[1]}"
        + f" --sheet-films {args.sheet_films[0]} {args.sheet_films[1]}"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
