"""Harmonic two-material conduction with an imperfect, massless thermal contact."""
import inspect
import json

import numpy as np
import torch

from conduction_provenance import provenance
from two_layer_conduction import slab_fem_model
from tensordem.thermal import contact_chain
from tensorfem import thermal
from tensorfvm.cht import energy


def reference(n, resistance, omega):
    """Propagate the continuum (temperature, rightward flux) from the cold wall."""
    waves = np.sqrt(1j*omega*np.array([1., 2.])/np.array([1., 4.]))

    def propagate(length, layer, temperature, flux):
        z = length*waves[layer]
        admittance = (1., 4.)[layer]*waves[layer]
        return (np.cosh(z)*temperature + np.sinh(z)*flux/admittance,
                admittance*np.sinh(z)*temperature + np.cosh(z)*flux)

    # Start with unit flux at x=1, then normalize the left boundary to one kelvin.
    cold_trace, contact_flux = propagate(.5, 1, 0., 1.)
    hot_trace = cold_trace + resistance*contact_flux
    scale, hot_flux = propagate(.5, 0, hot_trace, contact_flux)
    back = np.linspace(.5, 0., n//2+1)
    left, _ = propagate(back, 0, hot_trace, contact_flux)
    right, _ = propagate(back, 1, 0., 1.)
    nodes = np.r_[left, right]/scale
    # On a uniform cell the average is the midpoint value times sinh(z)/z.
    z = waves/(2*n)
    back_mid = .5*(back[:-1]+back[1:])
    avg = np.concatenate([propagate(back_mid, layer, temp, q)[0]*np.sinh(z[layer])/z[layer]
                          for layer, temp, q in ((0, hot_trace, contact_flux), (1, 0., 1.))])/scale
    return nodes, avg, contact_flux/scale, np.array([hot_flux, -1.])/scale


def measure(n, resistance):
    period = 1.
    omega = 2*np.pi/period
    time = np.linspace(0., period, 8*n+1)
    phase = np.exp(-1j*omega*time)
    dt = np.diff(time)
    mid_phase = np.exp(-1j*omega*(time[1:]+time[:-1])/2)
    m, dx = n//2, 1/n
    k, capacity = np.repeat([1., 4.], m), np.repeat([1., 2.], m)
    exact_nodes, avg, q, wall_ref = reference(n, resistance, omega)

    def harmonic(history):
        return (phase[:-1] @ history[:-1] + phase[1:] @ history[1:])/(len(time)-1)

    def record(name, cells, jump, interface, wall_phasor, left_input):
        storage = dx*capacity[0]*(cells[-1,:m]-cells[0,:m]).sum()
        left_residual = storage - (left_input - dt @ (.5*(interface[1:]+interface[:-1])))
        row = errors(name, harmonic(cells), harmonic(jump), harmonic(interface), wall_phasor)
        row["left_layer_balance_J_m2"] = float(abs(left_residual))
        return row

    def errors(name, temp, jump, interface, wall_phasor):
        return dict(solver=name, cells=n, resistance_m2K_W=resistance,
                    interface_phase_error_deg=float(abs(np.angle(interface/q, deg=True))),
                    temperature_error_K=float(np.max(abs(temp-avg))),
                    jump_error_K=float(abs(jump-resistance*q)),
                    interface_flux_error_W_m2=float(abs(interface-q)),
                    wall_flux_error_W_m2=float(np.max(abs(wall_phasor-wall_ref))))

    faces = np.zeros((1,n-1))
    faces[0,m-1] = resistance
    bc = {"west": {"type": "dirichlet", "values": lambda t: np.cos(omega*t)},
          "east": {"type": "dirichlet", "values": 0.}}
    t, powers = energy.solve_transient_conduction(np.full(n,dx), [1.], k, capacity, bc, time,
                    avg.real, contact_resistance_x=faces, return_wall_power=True)
    cells = t[:,0]
    interface = (cells[:,m-1]-cells[:,m])/(dx/(2*k[m-1])+resistance+dx/(2*k[m]))
    # Quadratic one-sided extrapolation of cell AVERAGES to each contact trace.
    weights = np.array([11., -7., 2.])/6
    jump = (cells[:,m-3:m][:,::-1]-cells[:,m:m+3]) @ weights
    walls = np.column_stack((powers["west"][:,0], powers["east"][:,0]))
    rows = [record("FVM", cells, jump, interface, harmonic(walls),
                   dt @ (.5*(walls[:-1,0]+walls[1:,0])))]

    phasor, wall, interior = energy.solve_harmonic_conduction(
        np.full(n,dx), [1.], k, capacity,
        {"west": {"type": "dirichlet", "values": 1.}, "east": {"type": "dirichlet", "values": 0.}},
        omega, contact_resistance_x=faces, return_interior_power=True)
    field = phasor[0]
    jump = (field[m-3:m][::-1]-field[m:m+3]) @ weights
    contact_power = interior["x"][0,m-1]
    wall_power = np.array([wall["west"][0], wall["east"][0]])
    row = errors("FVM_harmonic", field, jump, contact_power, wall_power)
    storage_rate = 1j*omega*dx*(capacity[:m] @ field[:m])
    row["left_layer_harmonic_balance_W_m2"] = float(abs(storage_rate - (wall_power[0]-contact_power)))
    rows.append(row)

    model, ends = slab_fem_model(k, 1., 0., resistance, None, torch.from_numpy(capacity))
    prescribed = np.column_stack((phase.real, phase.real, np.zeros(len(time)), np.zeros(len(time))))
    sol = thermal.solve_transient_thermal(model, torch.from_numpy(time),
                    torch.from_numpy(np.tile(exact_nodes.real, 2)), theta=.5, sparse=True,
                    fixed_temperature_history=torch.from_numpy(prescribed))
    traces = sol.temperature.numpy().reshape(len(time), 2, n+2).mean(axis=1)
    cells = np.concatenate([.5*(part[:,1:]+part[:,:-1]) for part in
                            (traces[:,:m+1], traces[:,m+1:])], axis=1)
    jump = traces[:,m] - traces[:,m+1]
    heat = sol.nodal_reaction_heat_steps()[:,ends].sum(dim=2).numpy()
    row = record("FEM", cells, jump, jump/resistance, 2/period*(mid_phase @ heat),
                 heat[:,0].sum())
    # Upper and lower edges are insulated. The two Q4 end nodes on each
    # material side carry the complete contact exchange for the unit area.
    local = thermal.thermal_element_nodal_heat_steps(model, sol).numpy()
    hot_out = -local[:, m-1, [1, 2]].sum(axis=1)
    cold_in = local[:, m, [0, 3]].sum(axis=1)
    link = .5*dt*(jump[:-1]+jump[1:])/resistance
    exact_heat = (q*np.diff(phase.conj())/(1j*omega)).real
    row["max_contact_mean_power_error_W_m2"] = float(abs((hot_out-exact_heat)/dt).max())
    row["max_contact_step_mismatch_J_m2"] = float(max(abs(hot_out-link).max(), abs(cold_in-link).max()))
    stored = dx*np.diff(cells[:,:m] @ capacity[:m])
    row["max_left_step_balance_J_m2"] = float(abs(stored-heat[:,0]+hot_out).max())
    rows.append(row)
    harmonic_sol = thermal.solve_harmonic_thermal(model, omega)
    traces = harmonic_sol.temperature.numpy().reshape(2, n+2).mean(axis=0)
    cells = np.concatenate([.5*(part[1:]+part[:-1]) for part in (traces[:m+1], traces[m+1:])])
    jump = traces[m] - traces[m+1]
    wall_powers = harmonic_sol.reaction_power[ends].sum(dim=1).numpy()
    row = errors("FEM_harmonic", cells, jump, jump/resistance, wall_powers)
    local_power = thermal.thermal_element_nodal_power(model, harmonic_sol.temperature, omega).numpy()
    hot_out = -local_power[m-1, [1, 2]].sum()
    cold_in = local_power[m, [0, 3]].sum()
    row["contact_element_flux_error_W_m2"] = float(max(abs(hot_out-q), abs(cold_in-q)))
    row["contact_power_mismatch_W_m2"] = float(max(abs(hot_out-jump/resistance), abs(cold_in-jump/resistance)))
    # Recover each material's Fourier flux from its own two nearest centers.
    # These diagnostics do not use the conductance of the contact link.
    axial = thermal.thermal_element_flux(model, harmonic_sol.temperature)[:,0].numpy()
    from_left = 1.5*axial[m-1] - .5*axial[m-2]
    from_right = 1.5*axial[m] - .5*axial[m+1]
    row["left_fourier_flux_error_W_m2"] = float(abs(from_left-q))
    row["right_fourier_flux_error_W_m2"] = float(abs(from_right-q))
    row["fourier_flux_mismatch_W_m2"] = float(abs(from_left-from_right))
    storage_rate = 1j*omega*dx*(capacity[:m] @ cells[:m])
    row["left_layer_harmonic_balance_W_m2"] = float(abs(storage_rate - (wall_powers[0]-jump/resistance)))
    rows.append(row)
    links, stores = contact_chain.cell_centered_slab_network(np.full(n, dx), k, capacity,
                                                              contact_resistance=faces[0])
    temperatures = contact_chain.harmonic_contact_temperatures(links, stores, omega, t_left=1., t_right=0.)
    field = temperatures[1:-1]
    link_flux = -np.diff(temperatures)/links
    jump = (field[m-3:m][::-1]-field[m:m+3]) @ weights
    row = errors("DEM", field, jump, link_flux[m], link_flux[[0,-1]]*[1.,-1.])
    storage_rate = 1j*omega*(stores[1:m+1] @ field[:m])
    row["left_layer_harmonic_balance_W_m2"] = float(abs(storage_rate-(link_flux[0]-link_flux[m])))
    rows.append(row)
    # Resolve the two interface traces as separate, massless nodes. The
    # adjacent cell halves and the areal contact resistance now form three links.
    resolved_r = np.r_[links[:m], dx/(2*k[m-1]), resistance, dx/(2*k[m]), links[m+1:]]
    resolved_c = np.insert(stores, m+1, [0., 0.])
    hot, cold = m+1, m+2
    cell_ids = np.flatnonzero(resolved_c)
    phasor = contact_chain.harmonic_contact_temperatures(resolved_r, resolved_c, omega,
                                       t_left=1., t_right=0., allow_massless=True)
    flux = -np.diff(phasor)/resolved_r
    row = errors("DEM_massless_harmonic", phasor[cell_ids], phasor[hot]-phasor[cold],
                 flux[hot], flux[[0,-1]]*[1.,-1.])
    row["trace_error_K"] = float(abs(phasor[[hot,cold]]-exact_nodes[m:m+2]).max())
    row["junction_flux_mismatch_W_m2"] = float(abs(np.diff(flux[m:m+3])).max())
    row["left_layer_harmonic_balance_W_m2"] = float(abs(
                1j*omega*(stores[1:m+1] @ phasor[cell_ids[:m]])-flux[0]+flux[hot]))
    rows.append(row)

    initial = np.zeros_like(resolved_c)
    initial[cell_ids] = avg.real
    history = contact_chain.transient_contact_temperatures(resolved_r, resolved_c, time, initial,
                                       t_left=phase.real, t_right=0., allow_massless=True)
    flux = -np.diff(history, axis=1)/resolved_r
    walls = flux[:, [0,-1]]*[1.,-1.]
    row = record("DEM_massless_transient", history[:,cell_ids], history[:,hot]-history[:,cold],
                 flux[:,hot], harmonic(walls), dt @ (.5*(walls[1:,0]+walls[:-1,0])))
    row["trace_error_K"] = float(abs(harmonic(history[:,[hot,cold]])-exact_nodes[m:m+2]).max())
    row["junction_flux_mismatch_W_m2"] = float(abs(np.diff(flux[:,m:m+3], axis=1)).max())
    rows.append(row)
    return rows


def main():
    rows = []
    for resistance in (.15, .6):
        previous = {}
        for n in (16, 32, 64, 128):
            for row in measure(n, resistance):
                name = row["solver"]
                errors = np.array([row[key] for key in ("temperature_error_K", "jump_error_K",
                                  "interface_flux_error_W_m2", "wall_flux_error_W_m2")])
                if name == "FEM_harmonic":
                    errors = np.append(errors, [row["left_fourier_flux_error_W_m2"],
                                                row["right_fourier_flux_error_W_m2"], row["contact_element_flux_error_W_m2"]])
                balance = (row["left_layer_harmonic_balance_W_m2"] if "left_layer_harmonic_balance_W_m2" in row
                           else row["left_layer_balance_J_m2"])
                passed = np.isfinite(errors).all() and balance < 1e-10
                if name == "FEM_harmonic":
                    passed &= row["contact_power_mismatch_W_m2"] < 1e-11
                if name == "FEM":
                    errors = np.append(errors, row["max_contact_mean_power_error_W_m2"])
                    passed &= max(row["max_contact_step_mismatch_J_m2"], row["max_left_step_balance_J_m2"]) < 1e-12
                if name.startswith("DEM_massless"):
                    errors = np.append(errors, row["trace_error_K"])
                    passed &= row["junction_flux_mismatch_W_m2"] < 1e-10
                if name in previous:
                    passed &= (errors < previous[name]/3).all()
                if n == 128:
                    passed &= (errors[:4] < [.001, .001, .002, .003]).all()
                    if len(errors) > 4:
                        passed &= (errors[4:] < .002).all()
                    passed &= row["interface_phase_error_deg"] < .1
                row["passed"] = bool(passed)
                previous[name] = errors
                rows.append(row)
    print(json.dumps({"sources": [provenance(f) for f in (__file__, inspect.getfile(slab_fem_model),
                      energy.__file__, thermal.__file__, contact_chain.__file__)], "period_s": 1., "runs": rows}, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
