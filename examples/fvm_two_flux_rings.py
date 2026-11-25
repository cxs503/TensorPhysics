"""Fixed inward heat-flux phasors at both walls of a three-material storing annulus."""
import json

import numpy as np

import bem_harmonic_storing_rings as geometry
import fvm_harmonic_storing_rings as analytic
from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(capacities, mode, omega):
    edges = np.array(geometry.RADII)[[0, 1, 3, 4]]
    walls = (None, None)
    flux = (.45-.25j, -.17+.31j)
    generation = np.array([.2j, -.3+.4j])
    fractions = (.2, .7)
    sheet_k = (.14, .35)
    exact, exact_storage, exact_wall = analytic.bulk_reference(
        edges, omega, mode, walls, generation, capacities, fractions, sheet_k, wall_fluxes=flux,
        return_bulk_storage=True, return_wall_temperatures=True)
    rows = []
    previous = None
    for n in (4, 8, 16, 32):
        mesh = np.r_[np.concatenate([np.geomspace(a,b,n+1)[:-1] for a,b in zip(edges, edges[1:])]), edges[-1]]
        sheet, qi, qo, rates, storage, lateral, surface = storing_rings.solve_storing_rings(
            mesh, np.repeat(geometry.K,n), [n,2*n], walls, omega, wall_fluxes=flux,
            volumetric_heat_capacity=np.repeat(capacities,n), contact_resistances=geometry.CONTACT,
            inner_resistance_fractions=fractions, sheet_capacities=geometry.CAPACITY,
            sheet_generation=generation, sheet_conductivities=sheet_k, azimuthal_mode=mode,
            return_wall_rates=True, return_bulk_storage=True, return_angular_conduction=True, return_wall_temperatures=True)
        ratio = surface/exact_wall
        wall_error = float(max(abs(ratio-1)))
        field_error = float(abs(np.column_stack((sheet, qi, qo))/exact-1).max())
        storage_error = float(max(abs(storage-exact_storage)))
        signed_rates = 2*np.pi*edges[[0,-1]]*np.array([flux[0], -flux[1]])
        mismatch = max(abs(rates-signed_rates))
        net = generation-(1j*omega*np.asarray(geometry.CAPACITY)+sheet_k*(mode/edges[1:-1])**2)*sheet
        balance = abs(signed_rates[0]-signed_rates[1]+2*np.pi*edges[1:-1]@net-sum(storage)-sum(lateral))
        error = max(wall_error, field_error)
        exact_radial = mode == 0 and not any(capacities)
        order = np.log2(previous/error) if previous is not None and not exact_radial else None
        resolved = error < 1e-10 if exact_radial else n < 32 or (error < .001 and order > 1.8)
        passed = resolved and balance < 1e-9 and mismatch < 1e-9 and (n < 32 or storage_error < .001)
        rows.append(dict(capacities_J_m3_K=capacities, mode=mode, omega=omega, cells_per_layer=n,
                         wall_relative_error=wall_error, sheet_field_relative_error=field_error, order=order,
                         wall_phase_error_deg=np.angle(ratio, deg=True).tolist(), storage_error_W_m=storage_error,
                         imposed_rate_residual_W_m=float(mismatch), modal_balance_W_m=float(balance), passed=bool(passed)))
        previous = error
    return rows


if __name__ == "__main__":
    runs = [r for cv in ((0.,0.,0.), (.7,2.3,1.2), (0.,2.3,0.)) for m in range(4)
            for w in (1.,8.,24.) for r in measure(cv,m,w)]
    passed = all(r["passed"] for r in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(p) for p in
        (__file__, analytic.__file__, geometry.__file__, annulus.__file__, storing_rings.__file__)],
        reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_two_flux_rings.py"), indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
