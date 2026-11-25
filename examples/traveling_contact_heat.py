"""Traveling thermal wave across two materials with a massless thermal contact."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
import periodic_contact_heat
from tensorfvm.cht import energy
from tensorsph import heat_conduction_2d as sph


def refine_layer_centers(coarse):
    """Cubic interpolation onto child centers without crossing the material interface."""
    m = len(coarse)//2
    target = (np.arange(2*m)+.5)/2-.5
    start = np.clip(np.floor(target).astype(int)-1, 0, m-4)
    nodes = start[:,None]+np.arange(4)
    weights = np.ones(nodes.shape)
    for i in range(4):
        for j in range(4):
            if i != j:
                weights[:,i] *= (target-nodes[:,j])/(i-j)
    return np.concatenate([(half[nodes]*weights).sum(axis=1) for half in np.split(coarse, 2)])


def measure(n, mode, resistance, sph_only=False):
    dx, m, omega = 1/n, n//2, 2*np.pi
    beta = 2*np.pi*mode
    phase = np.exp(1j*beta*(np.arange(n)+.5)*dx)
    phase_avg = phase*np.sinc(mode*dx)
    k, c = np.repeat([1., 4.], m), np.repeat([1., 2.], m)
    nodes, avg, q, walls = periodic_contact_heat.reference(n, resistance, omega, beta)
    z = .5*dx*np.sqrt(beta**2+1j*omega*c/k)
    point = avg*z/np.sinh(z)
    contact = np.zeros(n-1)
    contact[m-1] = resistance
    u,sections = sph.solve_harmonic_layered_slab(torch.zeros(n), dx, (1.,0.), omega,
                  conductivity=k, volumetric_heat_capacity=c, contact_resistance_x=contact,
                  transverse_wavenumber=beta,return_x_flux=True,backend="banded")
    left,right = sph.layered_slab_contact_temperatures(u,sections,dx,k)
    hot,cold = complex(left[m-1]),complex(right[m-1])
    traces = np.array([hot,cold])
    truth = nodes[m:m+2]
    harmonic = complex(sections[m])
    wall_error = abs(sections.numpy()[[0,-1]] - walls * [1.,-1.])
    yield harmonic,q,u.numpy(),point,dict(solver="SPH",n=n,mode=mode,contact_resistance_m2K_W=resistance,
        temperature_error_K=float(abs(u.numpy()-point).max()),interface_flux_error_W_m2=float(abs(harmonic-q)),
        interface_phase_error_rad=float(abs(np.angle(harmonic/q))),
        west_wall_flux_error_W_m2=float(wall_error[0]),east_wall_flux_error_W_m2=float(wall_error[1]),
        left_trace_error_K=float(abs(hot-truth[0])),right_trace_error_K=float(abs(cold-truth[1])),
        trace_phase_error_rad=float(abs(np.angle(traces*truth.conj())).max()),
        jump_error_K=float(abs(hot-cold-resistance*q)),contact_law_defect_K=float(abs(hot-cold-resistance*harmonic)))
    if sph_only:
        return
    widths = np.full(n, dx)
    fvm, _, interior = energy.solve_harmonic_conduction(widths, widths, k, c,
                    {"west": dict(type="dirichlet", values=phase_avg),
                     "east": dict(type="dirichlet", values=0.)}, omega, periodic_y=True,
                    contact_resistance_x=np.broadcast_to(contact, (n,n-1)), return_interior_power=True)
    measured = interior["x"][:,m-1]/dx
    harmonic = np.mean(measured/phase_avg)
    # Continue the half-cell temperature profiles to the material surface.
    # The FVM unknowns and normal flux use cell averages in y.
    surface = np.column_stack((fvm[:,m-1] - measured*dx/(2*k[m-1]),
                               fvm[:,m] + measured*dx/(2*k[m]))) / phase_avg[:,None]
    trace_error = abs(surface-truth).max(axis=0)
    jump = surface[:,0]-surface[:,1]
    yield harmonic,q,u.numpy(),point,dict(solver="FVM",n=n,mode=mode,contact_resistance_m2K_W=resistance,
        left_trace_error_K=float(trace_error[0]),right_trace_error_K=float(trace_error[1]),
        trace_phase_error_rad=float(abs(np.angle(surface*truth.conj())).max()),
        jump_error_K=float(abs(jump-resistance*q).max()),
        contact_law_defect_K=float(abs(jump-resistance*measured/phase_avg).max()),
        temperature_error_K=float(abs(fvm-phase_avg[:,None]*avg).max()),
        interface_flux_error_W_m2=float(abs(measured-phase_avg*q).max()),
        interface_phase_error_rad=float(abs(np.angle(measured/(phase_avg*q))).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sph-only",action="store_true",help="run only the banded SPH modal refinement")
    args = parser.parse_args()
    rows = []
    for resistance in (0., .15):
        previous = {}
        previous_traces = {}
        for n in (16,32,64,128,256,512):
            for harmonic, q, particle, point, row in measure(n, 1, resistance,args.sph_only):
                name = row["solver"]
                errors = np.array([row["temperature_error_K"], row["interface_flux_error_W_m2"]])
                orders = []
                extrap_error = temp_error = None
                if name in previous:
                    prior, prior_errors, prior_extrap, prior_particle, prior_temp = previous[name]
                    orders = np.log2(prior_errors/errors).tolist()
                    # Cancel the leading O(dx) error of the SPH contact flux.
                    if name == "SPH":
                        extrap = 2*harmonic-prior
                        temp_error = float(abs(2*particle-refine_layer_centers(prior_particle)-point).max())
                        row["extrapolated_temperature_error_K"] = temp_error
                        extrap_error = float(abs(extrap-q))
                        row["extrapolated_flux_error_W_m2"] = extrap_error
                        row["extrapolated_phase_error_rad"] = float(abs(np.angle(extrap/q)))
                        if prior_extrap is not None:
                            row["extrapolated_temperature_order"] = float(np.log2(prior_temp/temp_error))
                            row["extrapolated_flux_order"] = float(np.log2(prior_extrap/extrap_error))
                row["orders"] = orders
                row["raw_second_order_gate"] = all(p>1.8 for p in orders) if orders else None
                trace_keys = ("left_trace_error_K","right_trace_error_K","trace_phase_error_rad","jump_error_K")
                if name == "SPH":
                    trace_keys += ("west_wall_flux_error_W_m2","east_wall_flux_error_W_m2")
                if name in previous_traces:
                    row["trace_orders"] = {
                        k:float(np.log2(previous_traces[name][k]/row[k])) for k in trace_keys
                        if min(row[k],previous_traces[name][k]) > 1e-14
                    }
                previous_traces[name] = {k:row[k] for k in trace_keys}
                rows.append(row)
                previous[name] = harmonic, errors, extrap_error, particle, temp_error
    # Gate the finest refinement for each case. Retain raw orders at every resolution.
    finest = [r for r in rows if r["n"] == 512]
    for row in finest:
        if row["solver"] == "SPH":
            # The one-sided raw traces contain a first-order contact contribution.
            row["contact_trace_passed"] = (
                max(row["left_trace_error_K"],row["right_trace_error_K"]) < 1e-4
                and min(row["trace_orders"][side] for side in
                        ("left_trace_error_K","right_trace_error_K","jump_error_K")) > .85
                and row["jump_error_K"] < 1e-4 and row["contact_law_defect_K"] < 1e-6
            )
            temp_order = row["extrapolated_temperature_order"]
            temp_error = row["extrapolated_temperature_error_K"]
            flux_order = row["extrapolated_flux_order"]
            flux_error = row["extrapolated_flux_error_W_m2"]
        else:
            row["contact_trace_passed"] = (
                max(row["left_trace_error_K"],row["right_trace_error_K"]) < 1e-4
                and min(row["trace_orders"][side] for side in
                        ("left_trace_error_K","right_trace_error_K")) > 1.8
                and row["jump_error_K"] < 1e-4 and row["contact_law_defect_K"] < 1e-12
            )
            temp_order = row["orders"][0]
            temp_error = row["temperature_error_K"]
            flux_order = row["orders"][1]
            flux_error = row["interface_flux_error_W_m2"]
        row["acceptance_estimator"] = "two-grid extrapolated" if row["solver"] == "SPH" else "single-grid"
        row["passed"] = (min(temp_order, flux_order)>1.8 and flux_error<1e-4
                         and temp_error<2e-4 and row.get("contact_trace_passed",True))
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
                     (__file__, periodic_contact_heat.__file__, sph.__file__, energy.__file__)]),
                     indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in finest) else 1)
