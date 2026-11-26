"""Compare FEM and FVM interface power and directional cycle heat on a driven rod."""
from dataclasses import replace
import json
import math

import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import brentq

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import rod


def phase_integrals(bias, phasor, omega):
    """Independent integration of the positive and negative parts over one cycle."""
    phase = np.angle(phasor)
    roots = []
    if abs(bias) < abs(phasor):
        angle = math.acos(-bias/abs(phasor))
        roots = [(sign*angle-phase)%(2*math.pi) for sign in (-1,1)]
    def power(t):
        return bias+(phasor*np.exp(1j*t)).real
    heat = np.array([quad(lambda t: max(sign*power(t),0.),0.,2*math.pi,points=roots)[0]/omega
                     for sign in (1,-1)])
    return heat,np.sort(roots)/omega


def transient_contact_refinement(model,dc_model,dc,state,omega):
    """Time integration against the exact-in-time solution on this FEM mesh."""
    source = thermal.assemble_thermal(model,complex_loads=True).heat
    ac_power = thermal.thermal_contact_pair_power(model,state).item()
    dc_power = thermal.thermal_contact_pair_power(model,dc).item()
    initial = dc+state.real
    def ends(t):
        return dc_model.fixed_temperatures+(model.fixed_temperatures*torch.exp(1j*omega*t)).real
    def load(t):
        return (source*torch.exp(1j*omega*t)).real
    runs = []
    for method in ("theta","sdirk2"):
        previous = None
        for steps in (32,64,128):
            time = torch.linspace(0.,2*math.pi/omega,steps+1,dtype=dc.dtype)
            history = thermal.solve_transient_thermal(dc_model,time,initial,theta=.5,method=method,
                sparse=True,fixed_temperature_history=ends,nodal_heat_history=load)
            heat = thermal.thermal_contact_pair_heat_steps(dc_model,history)[:,0].cumsum(0)
            # Integrate the frequency-domain contact power from phase zero.
            exact_heat = dc_power*time+(ac_power*torch.expm1(1j*omega*time)/(1j*omega)).real
            exact_state = dc+(torch.exp(1j*omega*time[:,None])*state).real
            errors = dict(cumulative_heat_J=float((heat-exact_heat[1:]).abs().max()),
                          temperature_K=float((history.temperature-exact_state).abs().max()))
            row = dict(method="Crank-Nicolson" if method=="theta" else "SDIRK2",steps=steps,**errors)
            if previous is not None:
                for name,error in errors.items():
                    order = math.log2(previous[name]/error)
                    row[name+"_order"] = order
                    assert 1.8 < order < 2.2,(method,steps,name,order)
            runs.append(row)
            previous = errors
    return dict(reference="At 16 elements per layer, initialize from the steady field plus the real part of the same-mesh harmonic field. Compare all output times over one period against that field and its analytically integrated signed interface power. Errors are maxima over time (and nodes for temperature).",runs=runs)


def measure(n):
    omega, conductance = 2.3, 2.5
    contact, cold = .8-.4j, .7+.2j
    layers = [(2.,3.,1.,.75+.25j,cold+contact/conductance),
              (4.,1.5,.5,-.5+.75j,cold)]
    endpoints = []
    for i,(k,c,area,source,trace) in enumerate(layers):
        # k*T'' = i*omega*c*T-source. Start at the contact with known
        # temperature and rightward power, and propagate to the outer wall.
        wave = np.sqrt(1j*omega*c/k)
        z = (2*i-1)*wave
        base = source/(1j*omega*c)
        endpoints.append(base+(trace-base)*np.cosh(z)-contact*np.sinh(z)/(area*k*wave))
    k,c,areas,sources = [np.repeat([layer[i] for layer in layers],n) for i in range(4)]
    positions = np.r_[np.linspace(0.,1.,n+1),np.linspace(1.,2.,n+1)]
    starts = np.r_[np.arange(n),n+1+np.arange(n)]
    model = thermal.ThermalModel(torch.from_numpy(positions[:,None]),
        torch.from_numpy(np.column_stack((starts,starts+1))),torch.from_numpy(k),1.,torch.from_numpy(c),
        torch.tensor([0,2*n+1]),torch.tensor(endpoints,dtype=torch.complex128),
        thickness=torch.from_numpy(areas),heat_source=torch.from_numpy(sources),
        contact_pairs=torch.tensor([[n,n+1]]),contact_conductance=conductance)
    state = thermal.solve_harmonic_thermal(model,omega,sparse=True).temperature
    fem_power = thermal.thermal_contact_pair_power(model,state).item()
    # Both steady solves receive only wall temperatures. For the reference,
    # add the two unit-length layer resistances and the contact resistance.
    steady_ends = np.array([.49,0.])
    series_resistance = sum(1/(layer[0]*layer[2]) for layer in layers) + 1/conductance
    bias = (steady_ends[0]-steady_ends[1])/series_resistance
    dc_model = replace(model,fixed_temperatures=torch.from_numpy(steady_ends),heat_source=None)
    dc = thermal.solve_steady_thermal(dc_model,sparse=True)
    fem_bias = thermal.thermal_contact_pair_power(dc_model,dc).item()
    fem_cycle = np.array([v.item() for v in thermal.thermal_contact_pair_harmonic_cycle_heat(
        model,state,omega,steady_temperature=dc)])
    resistance = np.zeros(2*n-1)
    resistance[n-1] = min(areas)/conductance  # FVM takes areal resistance.
    widths = np.full(2*n,1/n)
    _,_,west,east = rod.solve_harmonic_stepped_rod(widths,areas,k,c,endpoints,omega,
        source=sources,contact_resistance=resistance,flux_order=2)
    fvm_power = east[n-1]
    _,_,_,steady_east = rod.solve_steady_stepped_rod(widths,areas,k,c,steady_ends,
        contact_resistance=resistance,flux_order=2)
    fvm_bias = steady_east[n-1]
    fvm_cycle = np.asarray(rod.harmonic_face_cycle_heat(fvm_power,omega,steady_power=fvm_bias))
    expected,exact_events = phase_integrals(bias,contact,omega)
    exact_duration = exact_events[1]-exact_events[0]
    period = 2*np.pi/omega
    rows = []
    for solver,power,dc_power,cycle in (("FEM",fem_power,fem_bias,fem_cycle),
                                        ("FVM",fvm_power,fvm_bias,fvm_cycle)):
        errors = abs(cycle-expected)
        def instantaneous_power(t):
            return dc_power+(power*np.exp(1j*omega*t)).real
        # This drive is forward at phase zero and reverse at half a cycle.
        # Locate each sign change from the solver's own DC and AC results.
        assert instantaneous_power(0)>0 and instantaneous_power(period/2)<0
        enter = brentq(instantaneous_power,0.,period/2,xtol=1e-14)
        leave = brentq(instantaneous_power,period/2,period,xtol=1e-14)
        rows.append(dict(solver=solver,elements_per_layer=n,power_error_W=float(abs(power-contact)),
                         steady_contact_power_W=float(dc_power),steady_power_error_W=float(abs(dc_power-bias)),
                         net_cycle_error_J=float(abs(cycle[0]-cycle[1]-2*np.pi*bias/omega)),
                         enter_reverse_s=enter,leave_reverse_s=leave,reverse_duration_s=leave-enter,
                         exact_reversal_times_s=exact_events.tolist(),exact_reverse_duration_s=exact_duration,
                         enter_error_s=abs(enter-exact_events[0]),leave_error_s=abs(leave-exact_events[1]),
                         duration_error_s=abs(leave-enter-exact_duration),
                         forward_error_J=float(errors[0]),reverse_error_J=float(errors[1]),
                         contact_power_W=[float(power.real),float(power.imag)]))
    if n==16:
        rows[0]["transient_refinement"] = transient_contact_refinement(model,dc_model,dc,state,omega)
    return rows,float(abs(fvm_power-west[n]))


if __name__ == "__main__":
    runs = []
    previous = {}
    for n in (8,16,32,64):
        rows,mismatch = measure(n)
        for row in rows:
            solver = row["solver"]
            assert row["steady_power_error_W"] < 1e-11 and row["net_cycle_error_J"] < 3e-11
            if solver=="FVM":
                row["interface_power_mismatch_W"] = mismatch
                assert mismatch < 1e-11
            if solver in previous:
                for metric in ("power_error_W","forward_error_J","reverse_error_J","enter_error_s","leave_error_s","duration_error_s"):
                    order = math.log2(previous[solver][metric]/row[metric])
                    row[metric+"_order"] = order
                    assert 1.8 < order < 2.2, (n,solver,metric,order)
            previous[solver] = row
            runs.append(row)
    print(json.dumps(dict(reference="Two unit-length layers with independent constant complex sources; continuum interface power 0.8-0.4i W, DC end temperatures 0.49 and 0 K in relative coordinates give 0.35 W through the series resistance 1.4 K/W, omega=2.3 rad/s. Integrate the signed halves of the biased sinusoid independently.",reproduce="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/stepped_rod_contact_cycle.py",benchmark=provenance(__file__),FEM=provenance(thermal.__file__),FVM=provenance(rod.__file__),runs=runs),indent=2,allow_nan=False))
