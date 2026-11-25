"""Oblique harmonic thermal wave in a slab with rotated principal conductivities."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
import q4_boundary_patches as patches
from tensorfem import thermal, thermal_analytic
from tensordg import poisson_2d_bc, transient_diffusion_2d


OMEGA, CAPACITY = 3., 1.5
WALL_STORAGE = {"right": .8, "top": .35}


def material(angle):
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    k = rotation @ np.diag([2.,.5]) @ rotation.T
    direction = np.array([1.,-.6])
    # T=exp(s·x) satisfies div(K grad T)=i omega c T.
    s = np.sqrt(1j*OMEGA*CAPACITY/(direction@k@direction))*direction
    return k,s


def dg_solution(n,degree,k,s,storing_walls=False):
    nx,ny = n,3*n//2
    def trace(x,y):
        return torch.exp(s[0]*x+s[1]*y)
    data = dict(conductivity=(k[0,0],k[1,1],k[0,1]),
                dirichlet_values=dict.fromkeys(("left","right","bottom","top"),trace))
    if storing_walls:
        # Right and top films supply k*n.grad(T) plus the local storage.
        gradient_flux = k@s
        films = {"right": 2.4, "top": 1.7}
        def ambient(side, axis):
            factor = 1+(gradient_flux[axis]+1j*OMEGA*WALL_STORAGE[side])/films[side]
            return lambda x,y: factor*trace(x,y)
        data.update(bc=dict(left="dirichlet", bottom="neumann", right="robin", top="robin"),
                    dirichlet_values={"left": trace},
                    neumann_flux={"bottom": lambda x,y: -gradient_flux[1]*trace(x,y)},
                    robin_films={side: (films[side], ambient(side, axis))
                                 for axis, side in enumerate(("right", "top"))},
                    wall_areal_heat_capacity=WALL_STORAGE)
    _,_,coeff = transient_diffusion_2d.solve_harmonic_diffusion_2d(
        OMEGA,ncells=(nx,ny),degree=degree,capacity=CAPACITY,**data)
    rate = dict(temperature_rate=1j*OMEGA*coeff) if storing_walls else {}
    fx,fy = (v.numpy() for v in poisson_2d_bc.poisson_face_powers_2d(coeff,**rate,**data))
    return coeff, fx, fy


def dg_case(n,degree,k,s,storing_walls=False,solution=None):
    nx, ny = n, 3*n//2
    x,y = np.linspace(0,1,nx+1),np.linspace(0,1,ny+1)
    coeff, fx, fy = dg_solution(n,degree,k,s,storing_walls) if solution is None else solution
    ex,ey = np.exp(s[0]*x),np.exp(s[1]*y)
    ax,ay = np.diff(ex)*nx/s[0],np.diff(ey)*ny/s[1]
    q = -k@s
    errors_x = abs(ny*fx-q[0]*np.outer(ex,ay))
    errors_y = abs(nx*fy-q[1]*np.outer(ax,ey))
    means = coeff[:,:,0,0].numpy()
    balance = 1j*OMEGA*CAPACITY*means/(nx*ny)+np.diff(fx,axis=0)+np.diff(fy,axis=1)
    wall_errors = {}
    if storing_walls:
        # At the positive face all normal Legendre modes are one; only
        # tangential mode zero contributes to the segment average.
        for side, numerical, exact in (("right", coeff[-1,:,:,0].sum(1).numpy(), ex[-1]*ay),
                                       ("top", coeff[:,-1,0,:].sum(1).numpy(), ax*ey[-1])):
            wall_errors[f"{side}_storage_error_W_m2"] = float(OMEGA*WALL_STORAGE[side]*abs(numerical-exact).max())
        for side, error in (("right", errors_x[-1]), ("top", errors_y[:, -1])):
            # All face segments on a wall have equal length. Use their centers
            # to define the same middle half of the physical wall on each grid.
            position = (np.arange(len(error))+.5)/len(error)
            middle = (.25 <= position) & (position < .75)
            wall_errors[f"{side}_bulk_flux_error_W_m2"] = float(error.max())
            wall_errors[f"{side}_rms_flux_error_W_m2"] = float(np.sqrt(np.mean(error**2)))
            wall_errors[f"{side}_middle_half_flux_error_W_m2"] = float(error[middle].max())
            wall_errors[f"{side}_peak_position_m"] = float(position[np.argmax(error)])
    return dict(**wall_errors,solver=f"DG-P{degree}",temperature_error_K=float(abs(means-np.outer(ax,ay)).max()),
                side_wall_error_W_m2=float(errors_x[[0,-1]].max()),
                end_wall_error_W_m2=float(errors_y[:,[0,-1]].max()),
                interior_flux_error_W_m2=float(max(errors_x[1:-1].max(),errors_y[:,1:-1].max())),
                max_balance_residual_W=float(abs(balance).max()))


def fem_solution(n,shear,k,s,warp):
    nx,ny = n,3*n//2
    mesh = thermal_analytic.structured_quad_mesh(nx,ny,shear=shear,dtype=torch.float64)
    coords = mesh.nodes
    if warp:
        x,y = coords.T
        displacement = warp*torch.sin(torch.pi*(x-shear*y))*torch.sin(torch.pi*y)
        coords = coords+torch.stack((displacement,-.6*displacement),dim=1)
    points = coords.numpy()
    ids = np.arange(len(points)).reshape(ny+1,nx+1)
    sides = [ids[:,0],ids[:,-1],ids[0,:],ids[-1,:]]
    fixed = np.unique(np.concatenate(sides))
    exact = np.exp(points@s)
    model = thermal.ThermalModel(coords,mesh.elements,torch.from_numpy(k[None]),1.,CAPACITY,
        torch.from_numpy(fixed),torch.from_numpy(exact[fixed]),element_type="q4")
    result = thermal.solve_harmonic_thermal(model,OMEGA,sparse=True)
    return model,result,sides,exact


def fem_case(n,shear,k,s,warp=0.):
    nx,ny = n,3*n//2
    model,result,sides,exact = fem_solution(n,shear,k,s,warp)
    coords = model.nodes
    points = coords.numpy()
    fixed = model.fixed_nodes.numpy()
    nodal = result.temperature.numpy()
    power = result.reaction_power.numpy()
    # Integrate the exact inward flux against the same boundary hats as
    # the FEM reaction. Each corner receives contributions from two sides.
    target = np.zeros_like(exact)
    area = np.zeros(len(points))
    unit,weights = np.polynomial.legendre.leggauss(8)
    shapes = np.column_stack(((1-unit)/2,(1+unit)/2))
    normals = [(-1,shear),(1,-shear),(0,-1),(0,1)]
    pairs = patches.boundary_pairs(nx,ny)
    raw = thermal.thermal_element_edge_power(model,result.temperature).numpy()
    raw_errors,edge_errors,patch_errors,sampled_errors,field_errors = [],[],[],[],[]
    cubic_errors,cubic_sampled,cubic_field,wide_quadratic = [],[],[],[]
    fields = [result.temperature,torch.from_numpy(exact)]
    refinement = {}
    if warp:
        fine_model,fine,_,_ = fem_solution(2*n,shear,k,s,warp)
        fine_shape = (2*ny+1,2*nx+1)
        nested = fine_model.nodes.reshape(*fine_shape,2)[::2,::2].reshape(-1,2)
        torch.testing.assert_close(nested,coords,rtol=0.,atol=1e-13)
        aligned = fine.temperature.reshape(fine_shape)[::2,::2].reshape(-1)
        extrapolated = aligned+(aligned-result.temperature)/3
        fields.append(extrapolated)
        refinement = dict(fine_nx=2*nx,fine_ny=2*ny,
            extrapolated_temperature_error_K=float(abs(extrapolated.numpy()-exact).max()))
    fields = torch.stack(fields)
    extrapolated_errors,fine_errors,extrapolated_powers = [],[],[]
    fine_pairs = {e:(o,i) for e,o,i in patches.boundary_pairs(2*nx,2*ny)} if warp else {}
    for edge,normal,(local,outer,inner) in zip(sides,normals,pairs):
        conn = np.column_stack((edge[:-1],edge[1:]))
        ends = points[conn]
        length = np.linalg.norm(ends[0,1]-ends[0,0])
        normal = np.array(normal)/np.linalg.norm(normal)
        values = np.exp((ends@s) @ shapes.T)*(normal@k@s)
        if not warp:
            outward = thermal.recover_q4_boundary_power(model,result.temperature,outer,inner,local)
            edge_errors.append(float(abs(outward.numpy()/length + values@weights/2).max()))
        along = np.arange(len(outer))+1
        along[-1] -= 2
        patch = np.column_stack((outer,inner,outer[along],inner[along]))
        fit = thermal.recover_q4_patch_boundary_power(model,fields,patch,local).numpy()
        err = abs(fit/length + values@weights/2).max(axis=1)
        patch_errors.append(float(err[0]))
        sampled_errors.append(float(err[1]))
        field_errors.append(float(abs(fit[0]-fit[1]).max()/length))
        wide = patches.cubic_patches(outer,inner)
        quadratic = thermal.recover_q4_patch_boundary_power(model,result.temperature,wide,local).numpy()
        wide_quadratic.append(float(abs(quadratic/length+values@weights/2).max()))
        cubic = thermal.recover_q4_patch_boundary_power(model,fields,wide,local,degree=3).numpy()
        cubic_err = abs(cubic/length+values@weights/2).max(axis=1)
        if warp:
            extrapolated_errors.append(float(cubic_err[2]))
            fine_patch = patches.cubic_patches(*fine_pairs[local])
            fine_power = thermal.recover_q4_patch_boundary_power(fine_model,fine.temperature,
                                                                  fine_patch,local,degree=3)
            # Two child edge powers sum to the power over the same physical edge.
            restricted = fine_power.numpy().reshape(-1,2).sum(1)
            fine_errors.append(float(abs(restricted/length+values@weights/2).max()))
            richardson = restricted+(restricted-cubic[0])/3
            extrapolated_powers.append(float(abs(richardson/length+values@weights/2).max()))
        cubic_errors.append(float(cubic_err[0]))
        cubic_sampled.append(float(cubic_err[1]))
        cubic_field.append(float(abs(cubic[0]-cubic[1]).max()/length))
        raw_errors.append(float(abs(raw[outer,local]/length + values@weights/2).max()))
        np.add.at(target,conn,(length/2)*(values*weights) @ shapes)
        np.add.at(area,conn,length/2)
    norm_error = abs(power[fixed]-target[fixed])/area[fixed]
    def side_error(edges):
        idx = np.concatenate([edge[1:-1] for edge in edges])
        return float((abs(power[idx]-target[idx])/area[idx]).max())
    capacity_matrix = thermal.assemble_thermal(model,sparse=True).capacity
    weights = capacity_matrix @ torch.ones_like(coords[:,0])
    storage = 1j*OMEGA*(weights.numpy()@nodal)
    prior = dict(recovered_edge_error_W_m2=max(edge_errors)) if edge_errors else {}
    if warp:
        refinement.update(extrapolated_cubic_error_W_m2=max(extrapolated_errors),
                          fine_cubic_error_W_m2=max(fine_errors),
                          extrapolated_edge_power_error_W_m2=max(extrapolated_powers))
    return dict(**refinement,**prior,solver="FEM-Q4",shear=shear,warp=warp,temperature_error_K=float(abs(nodal-exact).max()),
                side_wall_error_W_m2=side_error(sides[:2]),end_wall_error_W_m2=side_error(sides[2:]),
                cubic_edge_error_W_m2=max(cubic_errors),cubic_sampled_error_W_m2=max(cubic_sampled),
                cubic_field_contribution_W_m2=max(cubic_field),wide_quadratic_error_W_m2=max(wide_quadratic),
                raw_edge_error_W_m2=max(raw_errors),patch_edge_error_W_m2=max(patch_errors),sampled_patch_error_W_m2=max(sampled_errors),field_contribution_W_m2=max(field_errors),all_boundary_error_W_m2=float(norm_error.max()),
                max_balance_residual_W=float(abs(storage-power[fixed].sum())))


def compare_nested_flux(coarse, fine, k, s):
    coeff, fx, fy = coarse
    fine_coeff, fine_fx, fine_fy = fine
    nx, ny = coeff.shape[:2]
    # Sum child powers on each parent face, average temperatures over its four
    # children, then apply the same linear extrapolation to all three fields.
    restricted = (fine_coeff[:,:,0,0].numpy().reshape(nx,2,ny,2).mean(axis=(1,3)),
                  fine_fx[::2].reshape(nx+1,ny,2).sum(axis=2),
                  fine_fy[:,::2].reshape(nx,2,ny+1).sum(axis=1))
    extrapolated = tuple(f+(f-c)/3 for f,c in zip(restricted, (coeff[:,:,0,0].numpy(), fx, fy)))
    q = -k@s
    ax = np.diff(np.exp(s[0]*np.linspace(0,1,nx+1)))*nx/s[0]
    ay = np.diff(np.exp(s[1]*np.linspace(0,1,ny+1)))*ny/s[1]
    result = dict(coarse_nx=nx, coarse_ny=ny, assumed_order=2)
    for name, (mean, px, py) in (("restricted", restricted), ("extrapolated", extrapolated)):
        balance = 1j*OMEGA*CAPACITY*mean/(nx*ny)+np.diff(px,axis=0)+np.diff(py,axis=1)
        result[name] = dict(max_balance_residual_W=float(abs(balance).max()),
                            right_flux_error_W_m2=float(abs(ny*px[-1]-q[0]*np.exp(s[0])*ay).max()),
                            top_flux_error_W_m2=float(abs(nx*py[:,-1]-q[1]*np.exp(s[1])*ax).max()),
                            cell_mean_error_K=float(abs(mean-np.outer(ax,ay)).max()))
    return result


def verify_storing_walls():
    rows = []
    for angle in (-30., 30.):
        k, s = material(angle)
        for degree in (1, 2):
            previous = previous_solution = None
            for n in ((4, 8, 16, 32, 64, 128) if degree == 1 else (4, 8, 16, 32)):
                solution = dg_solution(n, degree, k, s, storing_walls=True)
                row = dg_case(n, degree, k, s, storing_walls=True, solution=solution)
                errors = {key: value for key, value in row.items() if "_error_" in key}
                orders = {} if previous is None else {
                    key: float(np.log2(previous[key]/value)) for key, value in errors.items()}
                row.update(nx=n, ny=3*n//2, angle_deg=angle, orders=orders,
                           passed=bool(row["max_balance_residual_W"] < 1e-10))
                if n >= 32:
                    wall_metrics = [key for key in errors if "_storage_error_" in key or "_bulk_flux_error_" in key]
                    row["wall_order_above_1p8"] = all(orders[key] > 1.8 for key in wall_metrics)
                    row["passed"] &= (errors["temperature_error_K"] < .001 and
                                      max(errors[key] for key in wall_metrics) < .01 and
                                      errors["interior_flux_error_W_m2"] < .04)
                    if n > 32:
                        row["passed"] &= all(orders[key] > 0 for key in errors if "flux_error_" in key)
                        for side in ("right", "top"):
                            row["passed"] &= orders[f"{side}_rms_flux_error_W_m2"] > 1.9
                            row["passed"] &= orders[f"{side}_middle_half_flux_error_W_m2"] > 1.9
                    if degree == 2:
                        row["passed"] &= row["wall_order_above_1p8"]
                if degree == 1 and previous_solution is not None:
                    comparison = compare_nested_flux(previous_solution, solution, k, s)
                    row["nested_comparison"] = comparison
                    fine, corrected = comparison["restricted"], comparison["extrapolated"]
                    row["passed"] &= max(fine["max_balance_residual_W"], corrected["max_balance_residual_W"]) < 1e-10
                    row["passed"] &= all(corrected[key] < fine[key] for key in
                                         ("right_flux_error_W_m2", "top_flux_error_W_m2", "cell_mean_error_K"))
                previous_solution = solution
                previous = errors
                rows.append(row)
    print(json.dumps(dict(reference="Oblique exponential thermal wave, left fixed temperature, bottom prescribed flux, right and top films with surface heat capacity",
                          reproduction="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorDG:../TensorFEM/src python examples/rotated_harmonic_conduction.py --dg-storing-walls",
                          runs=rows, sources=[provenance(f) for f in (__file__, transient_diffusion_2d.__file__, poisson_2d_bc.__file__)]),
                     indent=2, allow_nan=False))
    return all(row["passed"] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dg-storing-walls", action="store_true", help="verify DG with two storing films and mixed wall data")
    if parser.parse_args().dg_storing_walls:
        raise SystemExit(0 if verify_storing_walls() else 1)
    rows = []
    for angle in (-30.,30.):
        k,s = material(angle)
        previous = {}
        for n in (4,8,16,32,64):
            cases = [fem_case(n,.25,k,s,.1)]
            if n <= 32:
                cases.extend(dg_case(n,d,k,s) for d in (1,2))
                cases.extend(fem_case(n,h,k,s) for h in (0.,.25))
            for row in cases:
                name = row["solver"],row.get("shear",0.),row.get("warp",0.)
                keys = [key for key in row if "_error_" in key]
                if name in previous:
                    row["orders"] = {key:float(np.log2(previous[name][key]/row[key])) for key in keys}
                row.update(nx=n,ny=3*n//2,angle_deg=angle)
                row["passed"] = row["max_balance_residual_W"] < 1e-10
                if n == (64 if row.get("warp") else 32):
                    orders = row["orders"]
                    wall_order = .85 if name[0] == "DG-P1" else 1.8
                    wall_limit = {"DG-P1":1., "DG-P2":.006, "FEM-Q4":.002}[name[0]]
                    row["passed"] &= (row["temperature_error_K"] < .002
                                      and orders["temperature_error_K"] > 1.8
                                      and max(row["side_wall_error_W_m2"],row["end_wall_error_W_m2"]) < wall_limit
                                      and min(orders["side_wall_error_W_m2"],orders["end_wall_error_W_m2"]) > wall_order)
                    if name[0] == "FEM-Q4":
                        row["passed"] &= (row["patch_edge_error_W_m2"] < min(.025,row["raw_edge_error_W_m2"])
                                          and orders["patch_edge_error_W_m2"] > 1.8)
                        row["passed"] &= (row["cubic_edge_error_W_m2"] < min(.002,row["patch_edge_error_W_m2"],row["wide_quadratic_error_W_m2"])
                                          and orders["cubic_edge_error_W_m2"] > 1.8
                                          and orders["cubic_sampled_error_W_m2"] > 2.7)
                        if row["warp"]:
                            row["passed"] &= (row["extrapolated_temperature_error_K"] < 2e-8
                                              and row["extrapolated_edge_power_error_W_m2"] < min(3e-5,row["fine_cubic_error_W_m2"])
                                              and orders["extrapolated_edge_power_error_W_m2"] > 2.2
                                              and orders["extrapolated_cubic_error_W_m2"] > 2.7
                                              and row["extrapolated_cubic_error_W_m2"] < row["cubic_edge_error_W_m2"])
                        if not row["warp"]:
                            row["passed"] &= (row["recovered_edge_error_W_m2"] < min(.015,row["raw_edge_error_W_m2"])
                                          and orders["recovered_edge_error_W_m2"] > 1.8)
                        row["passed"] &= row["all_boundary_error_W_m2"] < .04 and orders["all_boundary_error_W_m2"] > .85
                    else:
                        row["passed"] &= orders["interior_flux_error_W_m2"] > wall_order
                previous[name] = row
                rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in
        (__file__,patches.__file__,thermal.__file__,thermal_analytic.__file__,transient_diffusion_2d.__file__,poisson_2d_bc.__file__)]),
        indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
