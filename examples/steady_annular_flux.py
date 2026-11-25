"""Steady closed-flux contact with a volume-mean (FVM) or bore-mean (BEM) datum."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorbem import layered_annulus
from tensorfvm.cht import annulus


def measure(n):
    a,b,c = .25,.6,1.
    k = np.array([2.,1.3])
    resistance, amplitude = .15,.2
    edges = np.r_[np.linspace(a,b,n+1),np.linspace(b,c,n+1)[1:]]
    centers = (2./3)*np.diff(edges**3)/np.diff(edges**2)
    contacts = np.zeros(2*n-1)
    contacts[n-1] = resistance
    ki,ko = k
    trace = amplitude*(1+2*resistance*ki/b)
    p,d = .5*(trace+amplitude*ki/ko), .5*(trace-amplitude*ki/ko)

    def solution(r, inner, mode):
        z = r/b
        if mode == 0:
            slope,offset = (1.,0.) if inner else (ki/ko,resistance*ki/b)
            return offset+slope*np.log(z), -slope*(ki if inner else ko)/r
        if inner:
            return amplitude*z**2, -2*ki*amplitude*z**2/r
        return p*z**2+d/z**2, -2*ko*(p*z**2-d/z**2)/r

    def integral_log(r):
        return r*r*(np.log(r/b)-.5)
    # Continuum volume mean; nonzero angular modes integrate to zero.
    mean = ((integral_log(b)-integral_log(a)) + ki/ko*(integral_log(c)-integral_log(b))
            + resistance*ki/b*(c*c-b*b))/(c*c-a*a)
    modes, reference, inputs, flux_errors = [], [], [], []
    for m in (0,2):
        bore = solution(a, True, m)[1]
        exterior = -solution(c, False, m)[1]
        t, rate = annulus.solve_annular_conduction(edges, np.repeat(k,n), 1., (None,None), 0.,
                     wall_fluxes=(bore, exterior), azimuthal_mode=m, contact_resistance=contacts,
                     cell_location="volume_centroid", distributed_source=True,
                     mean_temperature=mean if m == 0 else None)
        modes.append(t)
        reference.append(np.r_[solution(centers[:n],True,m)[0],solution(centers[n:],False,m)[0]])
        flux_errors.append(abs(rate[n]/(2*np.pi*b)-solution(b,True,m)[1]))
        inputs.append((bore,exterior))
    nt = 2*n
    angle = 2*np.pi*(np.arange(nt)+.5)/nt
    wave = np.cos(2*angle+.3)
    flux = np.asarray(inputs)[0,:,None] + np.asarray(inputs)[1,:,None]*wave
    bem = layered_annulus.solve_annulus_contact((a,b,c), k, (None,None), resistance, nt,
              formulation="circular", wall_fluxes=flux, mean_bore_temperature=np.log(a/b))
    # Use probe angles distinct from the BEM collocation grid.
    phi = .17+np.arange(2*n)*np.pi/n
    shape = np.cos(2*phi+.3)
    xy = centers[:,None]*np.stack((np.cos(phi),np.sin(phi)),axis=1)
    expected = reference[0]+reference[1]*shape
    bem_t = np.r_[bem.inner_field.temperature(xy[:n]),bem.outer_field.temperature(xy[n:])]
    exact_flux = solution(b,True,0)[1]+solution(b,True,2)[1]*wave
    return dict(radial_cells_per_layer=n, panels_per_ring=nt,
                fvm_temperature_error_K=float(np.max(abs(modes[0]+modes[1]*shape-expected))),
                fvm_contact_mode_error_W_m2=float(max(flux_errors)),
                bem_temperature_error_K=float(np.max(abs(bem_t-expected))),
                bem_contact_flux_error_W_m2=float(np.max(abs(bem.interface_flux-exact_flux))),
                fvm_mean_residual_K=float(abs(np.average(modes[0],weights=np.diff(edges**2))-mean)),
                bem_mean_residual_K=float(abs(bem.inner_field.boundary_values[nt:].mean()-np.log(a/b))))


def main():
    rows = []
    for n in (8,16,32,64):
        row = measure(n)
        errors = np.array([row["fvm_temperature_error_K"], row["fvm_contact_mode_error_W_m2"]])
        passed = max(row["bem_temperature_error_K"], row["bem_contact_flux_error_W_m2"]) < 1e-10
        passed &= max(row["fvm_mean_residual_K"],row["bem_mean_residual_K"]) < 1e-12
        if rows:
            passed &= (errors < previous/3.5).all()
        row["passed"] = bool(passed)
        previous = errors
        rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in (__file__,annulus.__file__,layered_annulus.__file__)],
                      "runs": rows},indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
