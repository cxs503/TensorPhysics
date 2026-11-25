"""Check two directly heated FVM rings against radial and angular continuum solutions."""
import argparse
import json

import numpy as np

import bem_harmonic_storing_rings as reference
from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def bulk_reference(edges, omega, mode, walls, generation, capacities, fractions=(.5, .5), sheet_k=(0., 0.), *, return_wall_rates=False):
    """Solve the six continuum coefficients using wall and sheet jump conditions."""
    from scipy.special import iv, ivp, kv, kvp

    def basis(layer, radius):
        wave = np.sqrt(1j*omega*capacities[layer]/reference.K[layer])
        z = wave*radius
        t, q = np.zeros((2, 6), dtype=complex)
        active = slice(2*layer, 2*layer+2)
        t[active] = [iv(mode, z), kv(mode, z)]
        q[active] = -reference.K[layer]*wave*np.array([ivp(mode, z), kvp(mode, z)])
        return t, q

    equations = [basis(0, edges[0])[0], basis(2, edges[-1])[0]]
    rhs = list(walls)
    samples = []
    for i, radius in enumerate(edges[1:-1]):
        left, incoming = basis(i, radius)
        right, outgoing = basis(i+1, radius)
        sheet = left - reference.CONTACT[i]*fractions[i]*incoming
        equations.extend([sheet - right - reference.CONTACT[i]*(1-fractions[i])*outgoing,
                          outgoing - incoming + (1j*omega*reference.CAPACITY[i]+sheet_k[i]*(mode/radius)**2)*sheet])
        rhs.extend([0., generation[i]])
        samples.append([sheet, incoming, outgoing])
    coeff = np.linalg.solve(equations, rhs)
    fields = np.asarray(samples) @ coeff
    if return_wall_rates:
        wall_flux = np.array([basis(0, edges[0])[1], basis(2, edges[-1])[1]]) @ coeff
        return fields, 2*np.pi*edges[[0, -1]]*wall_flux
    return fields


def measure(bulk_storage=False, fractions=(.5, .5), sheet_k=(0., 0.), wall_driven=False):
    edges = np.array(reference.RADII)[[0, 1, 3, 4]]
    walls = reference.WALLS if wall_driven else np.zeros(2)
    generation = np.zeros(2) if wall_driven else np.array([1., .6j])
    capacities = (0.7, 2.3, 1.2) if bulk_storage else None
    rows = []
    for mode in range(4):
        for omega in (1., 8., 24.):
            if bulk_storage:
                expected, wall_ref = bulk_reference(edges, omega, mode, walls, generation, capacities, fractions, sheet_k, return_wall_rates=True)
            elif mode:
                expected, wall_ref = reference.angular_reference(omega, mode, walls, generation, fractions, sheet_k, return_wall_rates=True)
            else:
                expected, wall_ref = reference.continuum(omega, walls, generation, fractions, return_wall_rates=True)
            previous = None
            previous_wall = None
            for n in (4, 8, 16, 32):
                mesh = np.r_[np.concatenate([np.geomspace(a, b, n+1)[:-1]
                                             for a, b in zip(edges, edges[1:])]), edges[-1]]
                temp, incoming, outgoing, wall_rates = storing_rings.solve_storing_rings(
                    mesh, np.repeat(reference.K, n), [n, 2*n], walls, omega,
                    contact_resistances=reference.CONTACT, sheet_capacities=reference.CAPACITY,
                    sheet_conductivities=sheet_k, inner_resistance_fractions=fractions, sheet_generation=generation, azimuthal_mode=mode,
                    volumetric_heat_capacity=np.repeat(capacities, n) if bulk_storage else None, return_wall_rates=True)
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
                rows.append(dict(wall_rate_max_relative_error=wall_error, wall_rate_order=wall_order,
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
    args = parser.parse_args()
    bulk_storage = args.bulk_storage
    fractions = args.inner_fractions
    runs = measure(bulk_storage, fractions, args.sheet_conductivities, args.wall_driven)
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, wall_driven=args.wall_driven, bulk_storage=bulk_storage, sheet_conductivities_W_K=args.sheet_conductivities,
                          inner_resistance_fractions=fractions, runs=runs, sources=[provenance(p) for p in
                          (__file__, reference.__file__, annulus.__file__, storing_rings.__file__)],
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_harmonic_storing_rings.py" + (" --bulk-storage" if bulk_storage else "")
                          + (" --wall-driven" if args.wall_driven else "")
                          + f" --inner-fractions {fractions[0]} {fractions[1]}"
                          + f" --sheet-conductivities {args.sheet_conductivities[0]} {args.sheet_conductivities[1]}"),
                     allow_nan=False, indent=2))
    raise SystemExit(0 if passed else 1)
