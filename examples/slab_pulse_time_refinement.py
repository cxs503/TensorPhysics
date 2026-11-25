"""Time refinement at fixed spatial resolution for a pulse entirely inside one reporting interval."""
import json
import math

import numpy as np
import torch
from tensorfem import thermal
from tensorfvm.cht import rod
from tensorsph import heat_conduction_2d as heat

import dg_subinterval_heat_pulse as dg
from conduction_provenance import provenance


def profiles(n, method, width, refinement):
    rate, amplitude = math.pi**2, 2**-44
    dt = 1/(16*rate)
    output = np.arange(19)*dt
    start = 1.2*dt
    stop = start+width*dt
    knots = np.union1d(output, [start, stop])
    # Bisect every smooth subinterval, including the short pulse, retaining
    # the original output intervals and precisely the same physical forcing.
    time = np.r_[np.concatenate([np.linspace(a,b,refinement+1)[:-1]
                                 for a,b in zip(knots,knots[1:])]), knots[-1]]
    duration = np.clip(time-start, 0, stop-start)
    ids = np.searchsorted(time, output)
    peak = amplitude*(-math.expm1(-rate*(stop-start)))
    def average(values):
        return np.add.reduceat(values*np.diff(time)[:, None], ids[:-1], axis=0)/(np.diff(output)[:, None]*peak)

    x = np.arange(n+1)/n
    centers = (np.arange(n)+0.5)/n
    mode = np.sin(math.pi*centers)
    scalar_source = amplitude*rate
    _, _, _, _, means = rod.solve_transient_stepped_rod(
        np.full(n, 1/n), 1., 1., 1., (305.,305.), time, np.full(n,305.), temperature_reference=305.,
        source=lambda t: scalar_source*np.sinc(.5/n)*mode if start <= t < stop else 0.,
        load_jump_times=(start,stop), method=method, flux_order=2, return_interval_temperatures=True)
    centered, _ = rod.reconstruct_interval_rod_field(np.full(n,1/n),1.,1.,means,[.5])
    source = torch.from_numpy(scalar_source*mode).expand(n,n)
    _, particles = heat.solve_transient_slab_2d(
        torch.full_like(source,305.), time, 1/n, (305.,305.), temperature_reference=305., method=method,
        heat_source=lambda t: source if start <= t < stop else 0., wall_jump_times=(start,stop),
        return_interval_temperatures=True)
    nodes = torch.from_numpy(x)
    elements = torch.stack((torch.arange(n),torch.arange(1,n+1)),dim=1)
    model = thermal.ThermalModel(nodes[:,None],elements,1.,1.,1.,torch.tensor([0,n]),nodes.new_zeros(2))
    # Exact integral of sin(pi*x) against each interior LINE2 hat function.
    load = np.sin(math.pi*x)*np.sinc(.5/n)**2/n
    load[[0,-1]] = (1-np.sinc(1/n))/math.pi
    fem = thermal.solve_transient_thermal(model, torch.from_numpy(time), torch.zeros_like(nodes), theta=.5,
        method=method, sparse=True, nodal_step_heat=torch.from_numpy(scalar_source*np.diff(duration)[:,None]*load))
    nodal = fem.interval_temperature.numpy()
    return {"fvm":average(centered[:,:,0]), "sph":average(particles.mean(dim=1).numpy()),
            "fem":average(.5*(nodal[:, :-1]+nodal[:, 1:]))}


if __name__ == "__main__":
    runs = []
    for width in (.05,.5):
        n = 64
        _, reference, _ = dg.measure(n,2,width,return_profiles=True)
        for method in ("theta","sdirk2"):
            last = None
            last_change = {}
            for subdivisions in (1,2,4,8):
                fields = profiles(n,method,width,subdivisions)
                for name,field in fields.items():
                    gap = float(abs(field-reference).max())
                    change = None if last is None else float(abs(field-last[name]).max())
                    order = None if name not in last_change else math.log2(last_change[name]/change)
                    passed = gap < .002 and (subdivisions < 8 or order > 1.9)
                    runs.append(dict(solver=name, method=method, width_over_interval=width, cells=n,
                                     subdivisions=subdivisions, max_difference_from_dg_over_peak=gap,
                                     time_refinement_change_over_peak=change, time_order=order, passed=passed))
                    if change is not None:
                        last_change[name] = change
                last = fields
    passed = all(row["passed"] for row in runs)
    print(json.dumps(dict(passed=passed, runs=runs, sources=[provenance(path) for path in
        (__file__, thermal.__file__, rod.__file__, heat.__file__, dg.__file__, dg.heat_1d.__file__)],
        reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src:../TensorSPH/src:../TensorDG python examples/slab_pulse_time_refinement.py"),
        indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
