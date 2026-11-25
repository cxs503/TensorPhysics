"""Compatible Neumann data for two unanchored sheets with radial volume generation."""
import argparse
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(n, omega, fractions, datum, compare_bem=False, mean_reference=False):
    edges = np.array([.4, .55, .9, 1.1])
    k = np.array([1., 2.5, .8])
    density = np.array([.6+.1j, -.3+.7j, .4-.2j])
    generation = np.array([.25-.4j, -.1+.3j])
    contact = np.array([.5, .9])
    # Integrate d(r q)/dr=r*g and dT/dr=-q/k in each material.
    bore_flux = .35+.12j
    rq = edges[0]*bore_flux
    integral = 0j
    t = datum
    sheets, incoming, outgoing, traces = [], [], [], []
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        constant = rq-density[i]*a*a/2
        # Integrate T against 2*r dr over this material before applying contact jumps.
        area = b*b-a*a
        integral += t*area - (constant*(b*b*np.log(b/a)-area/2) + density[i]*area**2/8)/k[i]
        t -= (constant*np.log(b/a)+density[i]*(b*b-a*a)/4)/k[i]
        rq = constant + density[i]*b*b/2
        if i < 2:
            qi = rq/b
            qo = qi+generation[i]
            sheet = t-fractions[i]*contact[i]*qi
            after = sheet-(1-fractions[i])*contact[i]*qo
            traces.append([t, after])
            incoming.append(qi)
            outgoing.append(qo)
            sheets.append(sheet)
            t, rq = after, b*qo
    mean = integral/(edges[-1]**2-edges[0]**2)
    expected_rates = 2*np.pi*np.array([edges[0]*bore_flux, rq])
    mesh = np.r_[np.concatenate([np.geomspace(a,b,n+1)[:-1] for a,b in zip(edges, edges[1:])]), edges[-1]]
    temp, qi, qo, rates, surfaces, contacts = storing_rings.solve_storing_rings(
        mesh, np.repeat(k,n), [n,2*n], (None,None), omega, wall_fluxes=(bore_flux, -rq/edges[-1]),
        bore_temperature_reference=None if mean_reference else datum,
        mean_temperature=mean if mean_reference else None, volumetric_heat_capacity=0. if omega else 1.,
        sheet_capacities=(0.,0.) if omega else (.25,.8), contact_resistances=contact,
        inner_resistance_fractions=fractions, source=np.repeat(density,n), sheet_generation=generation,
        return_wall_rates=True, return_wall_temperatures=True, return_contact_temperatures=True)
    t_error = max(abs(np.r_[temp-sheets, surfaces-[datum,t], (contacts-traces).ravel()]))
    q_error = max(abs(np.r_[qi-incoming, qo-outgoing]))
    rate_error = max(abs(rates-expected_rates))
    row = dict(cells_per_layer=n, omega=omega, inner_fractions=fractions, bore_temperature_K=[datum.real,datum.imag],
                max_temperature_error_K=float(t_error), max_flux_error_W_m2=float(q_error),
                max_wall_rate_error_W_m=float(rate_error), passed=bool(max(q_error,rate_error)<1e-10 and
                (t_error < 1e-10 if not mean_reference else t_error < .002/n**2)))
    if mean_reference:
        row["bulk_mean_temperature_K"] = [mean.real, mean.imag]
    if compare_bem:
        from tensorbem import harmonic_rings
        panels = 4*n
        response = harmonic_rings.assemble_ring_response(edges, k, contact, panels, flux_wall="both",
            formulation="circular", include_volume_sources=True, inner_resistance_fractions=fractions)
        bem = response.solve(omega, (None,None), (0.,0.) if omega else (.25,.8),
            wall_inward_flux=(bore_flux, -rq/edges[-1]), sheet_generation=generation,
            volume_generation=density, bore_temperature_reference=datum)
        temp_error = max(abs(np.r_[bem.sheet_temperature-sheets, bem.wall_surface_temperatures-[datum,t],
                                   (bem.contact_temperatures-traces).ravel()]))
        flux_error = max(abs(np.r_[bem.incoming_flux-incoming, bem.outgoing_flux-outgoing]))
        rate_error = max(abs(bem.wall_heat_rates-expected_rates))
        cross_temp = max(abs(np.r_[bem.sheet_temperature-temp, bem.wall_surface_temperatures-surfaces,
                                   (bem.contact_temperatures-contacts).ravel()]))
        row.update(bem_panels=panels, bem_temperature_error_K=float(temp_error), bem_flux_error_W_m2=float(flux_error),
                   bem_rate_error_W_m=float(rate_error), bem_join_residual_W_m=bem.join_power_residual,
                   solver_temperature_difference_K=float(cross_temp))
        row["passed"] &= bool(max(temp_error,flux_error,rate_error,bem.join_power_residual)<1e-10
                              and cross_temp < (.002/n**2 if mean_reference else 1e-10))
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-bem", action="store_true")
    parser.add_argument("--mean-reference", action="store_true", help="supply the continuum bulk volume mean to FVM")
    args = parser.parse_args()
    compare = args.compare_bem
    files = [__file__, storing_rings.__file__, annulus.__file__]
    path = "../TensorFVM/src"
    if compare:
        from tensorbem import harmonic_rings, layered_annulus
        files.extend([harmonic_rings.__file__, layered_annulus.__file__])
        path += ":../TensorBEM/src"
    runs = [measure(n,w,f,t,compare,args.mean_reference) for n in (2,8,32) for w in (0.,8.) for f in ((.2,.7),(0.,1.),(1.,0.))
            for t in (0.,1.3-.4j)]
    passed = all(x["passed"] for x in runs)
    print(json.dumps(dict(passed=passed,runs=runs,sources=[provenance(p) for p in files],
          reproduction=f"OPENBLAS_NUM_THREADS=1 PYTHONPATH={path} python examples/fvm_floating_annular_rings.py"
          + (" --compare-bem" if compare else "") + (" --mean-reference" if args.mean_reference else "")),indent=2,allow_nan=False))
    raise SystemExit(0 if passed else 1)
