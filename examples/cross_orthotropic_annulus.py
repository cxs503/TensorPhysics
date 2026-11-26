"""Verify cylindrical material orientation with Cartesian Q4 FEM and radial FVM."""
from dataclasses import replace
from functools import partial
import json

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import iv, ivp, kv, kvp

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import annulus, rod


def fem_model(radii, radial, circumferential, mode, wall_values):
    angles = 2*np.pi*np.arange(8*(len(radii)-1))/(8*(len(radii)-1))
    angular = np.cos(mode*angles + .37)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    nodes = (radii[:, None, None]*directions).reshape(-1, 2)
    ids = np.arange(len(nodes)).reshape(len(radii), len(angles))
    next_ids = np.roll(ids, -1, axis=1)
    conn = np.stack((ids[:-1], ids[1:], next_ids[1:], next_ids[:-1]), axis=-1).reshape(-1, 4)
    # Each Q4 has one material frame at its middle angle. Both its straight
    # boundary edges and its piecewise-constant orientation refine with n.
    middle = angles + np.pi/len(angles)
    direction = np.column_stack((np.cos(middle), np.sin(middle)))
    projector = direction[:, :, None]*direction[:, None, :]
    kr, kt = (np.broadcast_to(k, (len(radii)-1,))[:, None, None, None] for k in (radial, circumferential))
    tensors = kt*np.eye(2) + (kr-kt)*projector
    boundary = ids[[0, -1]].ravel()
    model = thermal.ThermalModel(
        torch.from_numpy(nodes), torch.from_numpy(conn), torch.from_numpy(tensors.reshape(-1, 2, 2)),
        1., 1., torch.from_numpy(boundary), torch.from_numpy((wall_values[:, None]*angular).ravel()),
        element_type="q4")
    return model, ids, angular


def measure(n):
    radial, circumferential, mode = 2., .7, 3
    power = mode*np.sqrt(circumferential/radial)
    radii = np.linspace(.4, 1., n+1)
    centers = .5*(radii[:-1]+radii[1:])
    wall_values = radii[[0, -1]]**power
    model, ids, angular = fem_model(radii, radial, circumferential, mode, wall_values)
    boundary = ids[[0, -1]].ravel()
    state = thermal.solve_steady_thermal(model, sparse=True)
    fem = state.numpy().reshape(ids.shape)
    matrices = thermal.assemble_thermal(model, sparse=True)
    reactions = (matrices.conductivity @ state - matrices.heat).numpy()[boundary].reshape(2, -1)
    # Nodal reactions are inward wall powers for unit depth. Integrating
    # against cos(m*theta+phase) gives pi*r*q; the FVM amplitude uses 2*pi*r*q.
    # Positive inward power at the outer wall points toward decreasing radius.
    fem_wall_rate = 2*(reactions @ angular)*np.array([1., -1.])
    fv, rates = annulus.solve_annular_conduction(radii, radial, 1., wall_values, 0.,
        azimuthal_mode=mode, circumferential_conductivity=circumferential, distributed_source=True)
    # Interpolate FEM along radial edges to compare at identical physical points.
    fem_mid = .5*(fem[1:] + fem[:-1])
    fv_mid = fv[:, None]*angular
    exact = centers[:, None]**power*angular
    exact_rate = -2*np.pi*radial*power*radii**power
    walls = {}
    for side, index, fem_rate in zip(("inner", "outer"), (0, -1), fem_wall_rate):
        errors = dict(fem=abs(fem_rate-exact_rate[index]), fvm=abs(rates[index]-exact_rate[index]),
                      difference=abs(fem_rate-rates[index]))
        for metric, error in errors.items():
            walls[f"{side}_{metric}_rate_error_W_per_m"] = float(error)
    return dict(**walls, radial_cells=n, sectors=len(angular),
                fem_error_K=float(np.max(abs(fem_mid-exact))),
                fvm_error_K=float(np.max(abs(fv_mid-exact))),
                solver_difference_K=float(np.max(abs(fem_mid-fv_mid))),
                fvm_modal_rate_error_W_per_m=float(np.max(abs(rates-exact_rate))))


def harmonic_wall_values(interface, walls, kr, order, omega, traces, rate):
    # The continuum solution in each material is A*I_order(w*r)+B*K_order(w*r),
    # with unit volumetric heat capacity and a prescribed interface flux.
    values = []
    for radius, k, nu, trace in zip(walls, kr, order, traces):
        w = np.sqrt(1j*omega/k)
        z = w*interface
        scale = -2*np.pi*interface*k*w
        coeff = np.linalg.solve([[iv(nu, z), kv(nu, z)],
                                 [scale*ivp(nu, z), scale*kvp(nu, z)]], [trace, rate])
        values.append(np.array([iv(nu, w*radius), kv(nu, w*radius)]) @ coeff)
    return np.array(values)


def contact_cycle_heat(model, state, fv_rates, exact_rate, omega):
    # In a period the positive lobe of a sinusoid integrates to 2/omega
    # times its peak. Around the ring the mean of |cos(m*theta+phi)| is
    # 2/pi for every nonzero integer mode. Rates already include 2*pi*r.
    factor = 4/(np.pi*omega)
    reference = factor*abs(exact_rate)
    fv_heat = factor*abs(fv_rates)
    fem_heat = thermal.thermal_contact_edge_harmonic_cycle_heat(model, state, omega).sum().item()
    net_phasor = thermal.thermal_contact_edge_power(model, state).sum().item()
    if abs(net_phasor) > 1e-10 or min(fem_heat, min(fv_heat)) <= 0:
        raise RuntimeError("Expected local cyclic exchange and zero integrated ring power")
    return dict(exact_each_direction_J_per_m=reference, fem_each_direction_J_per_m=fem_heat,
                fvm_each_direction_inner_J_per_m=float(fv_heat[0]),
                fvm_each_direction_outer_J_per_m=float(fv_heat[1]),
                fem_ring_net_power_amplitude_W_per_m=abs(net_phasor),
                fem_cycle_error_J_per_m=abs(fem_heat-reference),
                fvm_cycle_error_J_per_m=float(np.max(abs(fv_heat-reference))))


def biased_contact_cycle(model, state, radii, j, radial, resistance, fv_rate, exact_rate, omega):
    # Choose the DC wall temperatures from the continuum series resistance.
    bias = 1.5
    radius = radii[j]
    total_resistance = (np.log(radius/radii[0])/radial[0] + resistance/radius
                        + np.log(radii[-1]/radius)/radial[-1])/(2*np.pi)
    walls = np.array([bias*total_resistance, 0.])
    boundary = np.repeat(walls, len(model.fixed_nodes)//2)
    steady_model = replace(model, fixed_temperatures=torch.from_numpy(boundary))
    dc = thermal.solve_steady_thermal(steady_model, sparse=True)
    fem = np.array([heat.sum().item() for heat in thermal.thermal_contact_edge_biased_cycle_heat(
        model, state, omega, steady_temperature=dc)])
    _, inner, _, _ = annulus.solve_annular_sheet(radii, radial, 1., walls, 0., j, resistance=resistance)
    fv_bias = inner[-1].real
    # Spatial periodicity and time phase invariance reduce the angular mean to
    # a quadrant. Split where a local flux first ceases to reverse in time.
    limit = np.pi/2
    cutoff = np.arccos(np.clip(fv_bias/abs(fv_rate), 0., 1.))
    fv = np.array([quad(lambda a: rod.harmonic_face_cycle_heat(
        fv_rate*np.cos(a), omega, steady_power=fv_bias)[side], 0., limit,
        points=[cutoff], epsabs=1e-12, epsrel=1e-12)[0]/limit for side in (0, 1)])

    # Independent continuum reference: integrate instantaneous positive power
    # first around the circumference, then over time. Cosine symmetries apply
    # for every nonzero integer angular mode and any fixed phase shift.
    amplitude = abs(exact_rate)
    def spatial_mean(t, sign):
        peak = amplitude*np.cos(t)
        cuts = [np.arccos(-bias/peak)] if peak > bias else []
        return quad(lambda a: max(sign*(bias+peak*np.cos(a)), 0.), 0., np.pi,
                    points=cuts)[0]/np.pi
    exact = np.array([4/omega*quad(spatial_mean, 0., limit, args=(sign,),
        points=[np.arccos(bias/amplitude)], epsabs=1e-12, epsrel=1e-12)[0] for sign in (1, -1)])
    results = dict(biased_exact_outward_J_per_m=exact[0], biased_exact_inward_J_per_m=exact[1])
    for solver, measured, steady in (("fem", fem, thermal.thermal_contact_edge_power(model, dc).sum().item()),
                                      ("fvm", fv, fv_bias)):
        residual = abs(measured[0]-measured[1]-2*np.pi*steady/omega)
        if residual > 1e-10:
            raise RuntimeError(f"{solver} biased cycle energy residual {residual}")
        results[f"{solver}_biased_balance_J_per_m"] = residual
        for direction, value, reference in zip(("outward", "inward"), measured, exact):
            results[f"{solver}_biased_{direction}_error_J_per_m"] = abs(value-reference)
    return results


def measure_interface(n, resistance=0., omega=0.):
    mode, interface, temperature, rate = 3, .7, .8, 4.5
    radii = np.linspace(.4, 1., n+1)
    if omega:
        temperature, rate = .8-.2j, 4.5+1.5j
    j = n//2
    kr = np.array([2., .85])
    kt = np.array([.7, 2.2])
    # Within a homogeneous ring T is a combination of r^p and r^-p.
    # Extend each material-side temperature and the common radial rate from the
    # interface to obtain the two independent Dirichlet boundary amplitudes.
    power = mode*np.sqrt(kt/kr)
    argument = power*np.log(radii[[0, -1]]/interface)
    jump = resistance*rate/(2*np.pi*interface)
    exact_traces = np.array([temperature+jump, temperature])
    if omega:
        walls = harmonic_wall_values(interface, radii[[0, -1]], kr, power, omega, exact_traces, rate)
    else:
        walls = exact_traces*np.cosh(argument) - rate*np.sinh(argument)/(2*np.pi*kr*power)
    radial, circumferential = np.repeat(kr, j), np.repeat(kt, j)
    model, ids, angular = fem_model(radii, radial, circumferential, mode, walls)
    inner_nodes = torch.from_numpy(ids[j])
    outer_nodes = inner_nodes
    if resistance:
        # Give the outer material its own coincident interface nodes and use
        # the actual polygon edge length for the unit-depth contact area.
        outer_nodes = torch.arange(len(model.nodes), len(model.nodes)+len(angular))
        elements = model.elements.clone().reshape(n, len(angular), 4)
        elements[j, :, 0] = outer_nodes
        elements[j, :, 3] = outer_nodes.roll(-1)
        inner_edges = torch.stack((inner_nodes, inner_nodes.roll(-1)), dim=1)
        outer_edges = torch.stack((outer_nodes, outer_nodes.roll(-1)), dim=1)
        length = (model.nodes[inner_edges[:, 1]]-model.nodes[inner_edges[:, 0]]).norm(dim=1)
        model = replace(model, nodes=torch.cat((model.nodes, model.nodes[inner_nodes])),
                        elements=elements.reshape(-1, 4),
                        contact_edge_pairs=torch.stack((inner_edges, outer_edges), dim=1),
                        contact_edge_conductance=length/resistance)
    if omega:
        state = thermal.solve_harmonic_thermal(model, omega, sparse=True).temperature
    else:
        state = thermal.solve_steady_thermal(model, sparse=True)
    nodal = thermal.thermal_element_nodal_power(model, state, omega).numpy().reshape(n, len(angular), 4)
    # On a sector, local nodes 0 and 1 have its starting angle, 2 and 3
    # its ending angle. Recover the interface reaction from each side.
    following = np.roll(angular, -1)
    fem_rates = np.array([-2*(nodal[j-1, :, 1] @ angular + nodal[j-1, :, 2] @ following),
                           2*(nodal[j, :, 0] @ angular + nodal[j, :, 3] @ following)])
    _, inner, outer, _, fv_traces = annulus.solve_annular_sheet(
        radii, radial, 1., walls, omega, j, circumferential_conductivity=circumferential,
        azimuthal_mode=mode, distributed_source=True, resistance=resistance, return_contact_temperatures=True)
    fv_rates = np.array([inner[-1], outer[0]])
    balance = max(abs(np.diff(fem_rates)).item(), abs(np.diff(fv_rates)).item())
    if balance > 1e-9:
        raise RuntimeError(f"Interface rate imbalance {balance}")
    fem_traces = state[torch.stack((inner_nodes, outer_nodes))].numpy()
    jump_errors = {}
    if resistance:
        jump_errors = dict(fem_jump_error_K=float(np.max(abs(fem_traces[0]-fem_traces[1]-jump*angular))),
                           fvm_jump_error_K=float(abs(fv_traces[0]-fv_traces[1]-jump)))
    if omega and resistance:
        jump_errors.update(contact_cycle_heat(model, state, fv_rates, rate, omega))
        jump_errors.update(biased_contact_cycle(model, state, radii, j, radial, resistance, fv_rates[0], rate, omega))
        fem_jump = (fem_traces[0]-fem_traces[1]) @ angular / (angular @ angular)
        for solver, measured_jump, measured_rates in (
                ("fem", fem_jump, fem_rates), ("fvm", fv_traces[0]-fv_traces[1], fv_rates)):
            jump_errors[f"{solver}_jump_phase_error_rad"] = float(abs(np.angle(measured_jump/jump)))
            jump_errors[f"{solver}_rate_phase_error_rad"] = float(np.max(abs(np.angle(measured_rates/rate))))
    return dict(**jump_errors, radial_cells=n, sectors=len(angular), max_imbalance=float(balance),
                fem_interface_error_K=float(np.max(abs(fem_traces-exact_traces[:, None]*angular))),
                fvm_interface_error_K=float(np.max(abs(fv_traces-exact_traces))),
                fem_interface_rate_error_W_per_m=float(np.max(abs(fem_rates-rate))),
                fvm_interface_rate_error_W_per_m=float(np.max(abs(fv_rates-rate))),
                interface_rate_difference_W_per_m=float(np.max(abs(fem_rates-fv_rates))))


def refinement(case):
    runs = [case(n) for n in (8, 16, 32, 64)]
    for coarse, fine in zip(runs, runs[1:]):
        metrics = [key for key in fine if "error" in key or "difference" in key]
        for key in metrics:
            order = float(np.log2(coarse[key]/fine[key]))
            fine[key+"_order"] = order
            if fine["radial_cells"] >= 32 and order < 1.8:
                raise RuntimeError(f"{key} convergence order {order} on {fine['radial_cells']} cells")
    return runs


if __name__ == "__main__":
    print(json.dumps(dict(
        reference="Steady source-free unit-depth annulus, T=r^p*cos(3*theta+0.37), p=3*sqrt(k_theta/k_r), k_r=2 and k_theta=0.7 W/(m K). Dirichlet walls at 0.4 and 1 m. Maximum temperature errors at shared radial-edge midpoints. Wall rates are 2*pi*r times increasing-radius flux amplitude, recovered from FEM nodal reaction powers by cosine projection and from FVM boundary faces directly.",
        reproduce="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/cross_orthotropic_annulus.py",
        benchmark=provenance(__file__), FEM=provenance(thermal.__file__), FVM=provenance(annulus.__file__), FVM_cycle=provenance(rod.__file__),
        interface_reference="Perfect contact at r=0.7 m, T amplitude 0.8 K and outward radial rate amplitude 4.5 W/m. Inner (radial, circumferential) conductivity (2, 0.7), outer (0.85, 2.2) W/(m K). Set wall temperatures from the two independent power-law solutions. max_imbalance is the largest inner/outer rate discrepancy in W/m.",
        contact_reference="Same two materials with areal resistance 0.25 m² K/W. The outer trace amplitude is 0.8 K, inner minus outer equals R*4.5/(2*pi*0.7) K. FEM contact edges use their chord lengths; FVM uses the circular interface.",
        harmonic_reference="Same annulus and contact, unit volumetric heat capacity, omega=6 rad/s, exp(+i*omega*t). Outer trace 0.8-0.2i K, transmitted rate 4.5+1.5i W/m. Derive exterior Dirichlet phasors from independent modified Bessel solutions in each material. Errors in K and W/m are complex absolute errors; phase errors are wrapped differences in radians. For unit axial depth, local forward and reverse heat per period each integrate to 4*abs(rate)/(pi*omega) J/m around the ring. The circumferentially integrated signed power vanishes at every instant.",
        biased_reference="Superpose an axisymmetric DC rate of 1.5 W/m using the two log radial resistances and finite contact resistance. Integrate instantaneous continuum flux over angle then phase; compare to FEM edge cycles and FVM angular integration of local periodic heat.",
        harmonic_runs=refinement(partial(measure_interface, resistance=.25, omega=6.)),
        contact_runs=refinement(partial(measure_interface, resistance=.25)),
        runs=refinement(measure), interface_runs=refinement(measure_interface)), indent=2, allow_nan=False))
