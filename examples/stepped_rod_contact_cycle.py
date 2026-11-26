"""Compare FEM and FVM interface power and directional cycle heat on a driven rod."""
from dataclasses import replace
import json
import math

import numpy as np
import torch
from scipy.integrate import quad

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
    return np.array([quad(lambda t: max(sign*power(t),0.),0.,2*math.pi,points=roots)[0]/omega
                     for sign in (1,-1)])


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
    expected = phase_integrals(bias,contact,omega)
    rows = []
    for solver,power,dc_power,cycle in (("FEM",fem_power,fem_bias,fem_cycle),
                                        ("FVM",fvm_power,fvm_bias,fvm_cycle)):
        errors = abs(cycle-expected)
        rows.append(dict(solver=solver,elements_per_layer=n,power_error_W=float(abs(power-contact)),
                         steady_contact_power_W=float(dc_power),steady_power_error_W=float(abs(dc_power-bias)),
                         net_cycle_error_J=float(abs(cycle[0]-cycle[1]-2*np.pi*bias/omega)),
                         forward_error_J=float(errors[0]),reverse_error_J=float(errors[1]),
                         contact_power_W=[float(power.real),float(power.imag)]))
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
                for metric in ("power_error_W","forward_error_J","reverse_error_J"):
                    order = math.log2(previous[solver][metric]/row[metric])
                    row[metric+"_order"] = order
                    assert 1.8 < order < 2.2, (n,solver,metric,order)
            previous[solver] = row
            runs.append(row)
    print(json.dumps(dict(reference="Two unit-length layers with independent constant complex sources; continuum interface power 0.8-0.4i W, DC end temperatures 0.49 and 0 K in relative coordinates give 0.35 W through the series resistance 1.4 K/W, omega=2.3 rad/s. Integrate the signed halves of the biased sinusoid independently.",reproduce="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/stepped_rod_contact_cycle.py",benchmark=provenance(__file__),FEM=provenance(thermal.__file__),FVM=provenance(rod.__file__),runs=runs),indent=2,allow_nan=False))
