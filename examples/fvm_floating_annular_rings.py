"""Compatible Neumann data for two unanchored sheets with radial volume generation."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import annulus, storing_rings


def measure(n, omega, fractions, datum):
    edges = np.array([.4, .55, .9, 1.1])
    k = np.array([1., 2.5, .8])
    density = np.array([.6+.1j, -.3+.7j, .4-.2j])
    generation = np.array([.25-.4j, -.1+.3j])
    contact = np.array([.5, .9])
    # Integrate d(r q)/dr=r*g and dT/dr=-q/k in each material.
    bore_flux = .35+.12j
    rq = edges[0]*bore_flux
    t = datum
    sheets, incoming, outgoing, traces = [], [], [], []
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        constant = rq-density[i]*a*a/2
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
    expected_rates = 2*np.pi*np.array([edges[0]*bore_flux, rq])
    mesh = np.r_[np.concatenate([np.geomspace(a,b,n+1)[:-1] for a,b in zip(edges, edges[1:])]), edges[-1]]
    temp, qi, qo, rates, surfaces, contacts = storing_rings.solve_storing_rings(
        mesh, np.repeat(k,n), [n,2*n], (None,None), omega, wall_fluxes=(bore_flux, -rq/edges[-1]),
        bore_temperature_reference=datum, volumetric_heat_capacity=0. if omega else 1.,
        sheet_capacities=(0.,0.) if omega else (.25,.8), contact_resistances=contact,
        inner_resistance_fractions=fractions, source=np.repeat(density,n), sheet_generation=generation,
        return_wall_rates=True, return_wall_temperatures=True, return_contact_temperatures=True)
    t_error = max(abs(np.r_[temp-sheets, surfaces-[datum,t], (contacts-traces).ravel()]))
    q_error = max(abs(np.r_[qi-incoming, qo-outgoing]))
    rate_error = max(abs(rates-expected_rates))
    return dict(cells_per_layer=n, omega=omega, inner_fractions=fractions, bore_temperature_K=[datum.real,datum.imag],
                max_temperature_error_K=float(t_error), max_flux_error_W_m2=float(q_error),
                max_wall_rate_error_W_m=float(rate_error), passed=bool(max(t_error,q_error,rate_error)<1e-10))


if __name__ == "__main__":
    runs = [measure(n,w,f,t) for n in (2,8,32) for w in (0.,8.) for f in ((.2,.7),(0.,1.),(1.,0.))
            for t in (0.,1.3-.4j)]
    passed = all(x["passed"] for x in runs)
    print(json.dumps(dict(passed=passed,runs=runs,sources=[provenance(p) for p in
          (__file__, storing_rings.__file__, annulus.__file__)],
          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_floating_annular_rings.py"),indent=2,allow_nan=False))
    raise SystemExit(0 if passed else 1)
