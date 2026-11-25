"""Check two directly heated FVM rings against radial and angular continuum solutions."""
import argparse
import json

import numpy as np

import bem_harmonic_storing_rings as reference
from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def bulk_reference(edges, omega, mode, walls, generation, capacities, fractions=(.5, .5), sheet_k=(0., 0.), *,
                   return_wall_rates=False, return_bulk_storage=False, films=(None, None), return_contact_temperatures=False):
    """Solve the six continuum coefficients using wall and sheet jump conditions."""
    from scipy.special import iv, ivp, kv, kvp

    def basis(layer, radius):
        wave = np.sqrt(1j*omega*capacities[layer]/reference.K[layer])
        z = wave*radius
        t, q = np.zeros((2, 6), dtype=complex)
        active = slice(2*layer, 2*layer+2)
        if wave == 0:
            if mode:
                t[active] = [radius**mode, radius**(-mode)]
                q[active] = -reference.K[layer]*mode/radius*np.array([radius**mode, -radius**(-mode)])
            else:
                t[active] = [1., np.log(radius)]
                q[active] = [0., -reference.K[layer]/radius]
            return t, q
        t[active] = [iv(mode, z), kv(mode, z)]
        q[active] = -reference.K[layer]*wave*np.array([ivp(mode, z), kvp(mode, z)])
        return t, q

    bore, bore_flux = basis(0, edges[0])
    exterior, exterior_flux = basis(2, edges[-1])
    equations = [bore if films[0] is None else bore+bore_flux/films[0],
                 exterior if films[1] is None else exterior-exterior_flux/films[1]]
    rhs = list(walls)
    samples = []
    traces = []
    for i, radius in enumerate(edges[1:-1]):
        left, incoming = basis(i, radius)
        right, outgoing = basis(i+1, radius)
        sheet = left - reference.CONTACT[i]*fractions[i]*incoming
        equations.extend([sheet - right - reference.CONTACT[i]*(1-fractions[i])*outgoing,
                          outgoing - incoming + (1j*omega*reference.CAPACITY[i]+sheet_k[i]*(mode/radius)**2)*sheet])
        rhs.extend([0., generation[i]])
        samples.append([sheet, incoming, outgoing])
        traces.append([left, right])
    coeff = np.linalg.solve(equations, rhs)
    fields = np.asarray(samples) @ coeff
    result = (fields,)
    if return_wall_rates:
        wall_flux = np.array([bore_flux, exterior_flux]) @ coeff
        result += (2*np.pi*edges[[0, -1]]*wall_flux,)
    if return_bulk_storage:
        from scipy.integrate import quad_vec
        rates = []
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            integral, _ = quad_vec(lambda r: r*(basis(i, r)[0] @ coeff), a, b,
                                   epsabs=1e-13, epsrel=1e-12)
            rates.append(2j*np.pi*omega*capacities[i]*integral)
        result += (np.asarray(rates),)
    if return_contact_temperatures:
        result += (np.asarray(traces) @ coeff,)
    return result if len(result) > 1 else fields


def measure(bulk_storage=False, fractions=(.5, .5), sheet_k=(0., 0.), wall_driven=False, films=(None, None)):
    edges = np.array(reference.RADII)[[0, 1, 3, 4]]
    walls = reference.WALLS if wall_driven else np.zeros(2)
    generation = np.zeros(2) if wall_driven else np.array([1., .6j])
    capacities = (0.7, 2.3, 1.2) if bulk_storage else None
    rows = []
    for mode in range(4):
        for omega in (1., 8., 24.):
            if bulk_storage:
                expected, wall_ref, storage_ref = bulk_reference(edges, omega, mode, walls, generation, capacities, fractions, sheet_k, return_wall_rates=True, return_bulk_storage=True)
            elif mode:
                expected, wall_ref = reference.angular_reference(omega, mode, walls, generation, fractions, sheet_k, return_wall_rates=True)
            else:
                expected, wall_ref = reference.continuum(omega, walls, generation, fractions, return_wall_rates=True)
            # For increasing-r heat rates, the ambient shift has opposite signs
            # at bore and exterior. Use the exact solution to prescribe the loads.
            boundary = walls.astype(complex).copy()
            for j, (h, radius, sign) in enumerate(zip(films, edges[[0, -1]], (1, -1))):
                if h is not None:
                    boundary[j] += sign*wall_ref[j]/(2*np.pi*radius*h)
            area = 2*np.pi*edges[1:-1]
            # Continuum angular divergence follows from the analytic radial
            # rates on either side of each layer and its integrated storage.
            reference_radial_drop = (np.r_[wall_ref[0], area*expected[:, 2]]
                                     - np.r_[area*expected[:, 1], wall_ref[1]])
            angular_ref = reference_radial_drop - (storage_ref if bulk_storage else 0.)
            previous_angular = None
            previous_storage = None
            previous = None
            previous_wall = None
            for n in (4, 8, 16, 32):
                mesh = np.r_[np.concatenate([np.geomspace(a, b, n+1)[:-1]
                                             for a, b in zip(edges, edges[1:])]), edges[-1]]
                temp, incoming, outgoing, wall_rates, storage, angular = storing_rings.solve_storing_rings(
                    mesh, np.repeat(reference.K, n), [n, 2*n], boundary, omega, film_coefficients=films,
                    contact_resistances=reference.CONTACT, sheet_capacities=reference.CAPACITY,
                    sheet_conductivities=sheet_k, inner_resistance_fractions=fractions, sheet_generation=generation, azimuthal_mode=mode,
                    volumetric_heat_capacity=np.repeat(capacities, n) if bulk_storage else None,
                    return_wall_rates=True, return_bulk_storage=True, return_angular_conduction=True)
                wall_error = float(abs(wall_rates/wall_ref-1).max())
                angular_loss = sheet_k*(mode/edges[1:-1])**2*temp
                balance = max(abs(incoming-outgoing+generation-angular_loss-1j*omega*np.asarray(reference.CAPACITY)*temp))
                error = float(abs(np.column_stack((temp, incoming, outgoing))/expected-1).max())
                order = np.log2(previous/error) if previous is not None and (mode or bulk_storage) else None
                passed = balance < 1e-10 and (error < 1e-10 if mode == 0 and not bulk_storage else
                                               n < 32 or (error < .001 and order > 1.8))
                wall_order = np.log2(previous_wall/wall_error) if previous_wall is not None and (mode or bulk_storage) else None
                passed &= (wall_error < 1e-10 if mode == 0 and not bulk_storage else
                           n < 32 or (wall_error < .001 and wall_order > 1.8))
                storage_error = float(abs(storage/storage_ref-1).max()) if bulk_storage else None
                storage_order = np.log2(previous_storage/storage_error) if previous_storage is not None else None
                passed &= (n < 32 or (storage_error < .001 and storage_order > 1.8)) if bulk_storage else np.all(storage == 0.)
                previous_storage = storage_error
                layer_input = np.r_[wall_rates[0], area*outgoing] - np.r_[area*incoming, wall_rates[1]]
                layer_balance = float(max(abs(layer_input - storage - angular)))
                net_sheet = area @ (generation-angular_loss-1j*omega*np.asarray(reference.CAPACITY)*temp)
                global_balance = float(abs(wall_rates[0]-wall_rates[1]+net_sheet-storage.sum()-angular.sum()))
                angular_error = float(max(abs(angular/angular_ref-1))) if mode else None
                angular_order = np.log2(previous_angular/angular_error) if previous_angular is not None else None
                previous_angular = angular_error
                passed &= layer_balance < 1e-9 and global_balance < 1e-9
                passed &= (n < 32 or (angular_error < .001 and angular_order > 1.8)) if mode else np.all(angular == 0.)
                rows.append(dict(bulk_storage_rates_W_m=[[z.real, z.imag] for z in storage],
                                 bulk_storage_max_relative_error=storage_error, bulk_storage_order=storage_order,
                                 axisymmetric_balance_W_m=global_balance if mode == 0 else None,
                                 angular_conduction_W_m=[[z.real, z.imag] for z in angular],
                                 angular_max_relative_error=angular_error, angular_order=angular_order,
                                 max_layer_balance_W_m=layer_balance, modal_global_balance_W_m=global_balance,
                                 wall_rate_max_relative_error=wall_error, wall_rate_order=wall_order,
                                 radial_wall_rates_W_m=[[z.real, z.imag] for z in wall_rates],
                                 mode=mode, omega=omega, cells_per_layer=n, max_relative_error=error,
                                 order=order, sheet_balance_W_m2=float(balance), passed=bool(passed)))
                previous = error
                previous_wall = wall_error
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-storage", action="store_true", help="include volumetric heat storage in all three layers")
    parser.add_argument("--inner-fractions", nargs=2, type=float, default=(.5, .5),
                        metavar=("RING1", "RING2"), help="fraction of each contact resistance on its inner side")
    parser.add_argument("--sheet-conductivities", nargs=2, type=float, default=(0., 0.),
                        metavar=("RING1", "RING2"), help="circumferential conductivity times thickness in W/K")
    parser.add_argument("--wall-driven", action="store_true", help="apply wall temperature phasors with zero sheet generation")
    parser.add_argument("--film-walls", choices=("none", "bore", "exterior", "both"), default="none",
                        help="impose ambient temperatures through bore/exterior films of 7/3 W/(m² K)")
    args = parser.parse_args()
    films = (7. if args.film_walls in ("bore", "both") else None,
             3. if args.film_walls in ("exterior", "both") else None)
    bulk_storage = args.bulk_storage
    fractions = args.inner_fractions
    runs = measure(bulk_storage, fractions, args.sheet_conductivities, args.wall_driven, films)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, film_coefficients_W_m2_K=films, wall_driven=args.wall_driven, bulk_storage=bulk_storage, sheet_conductivities_W_K=args.sheet_conductivities,
                          inner_resistance_fractions=fractions, runs=runs, sources=[provenance(p) for p in
                          (__file__, reference.__file__, annulus.__file__, storing_rings.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_harmonic_storing_rings.py" + (" --bulk-storage" if bulk_storage else "")
                          + (" --wall-driven" if args.wall_driven else "")
                          + f" --film-walls {args.film_walls}"
                          + f" --inner-fractions {fractions[0]} {fractions[1]}"
                          + f" --sheet-conductivities {args.sheet_conductivities[0]} {args.sheet_conductivities[1]}"),
                     allow_nan=False, indent=2))
    raise SystemExit(0 if passed else 1)
