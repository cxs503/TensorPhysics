"""Verify cylindrical material orientation with Cartesian Q4 FEM and radial FVM."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfem import thermal
from tensorfvm.cht import annulus


def measure(n):
    radial, circumferential, mode = 2., .7, 3
    power = mode*np.sqrt(circumferential/radial)
    radii = np.linspace(.4, 1., n+1)
    centers = .5*(radii[:-1]+radii[1:])
    angles = 2*np.pi*np.arange(8*n)/(8*n)
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
    tensors = circumferential*np.eye(2) + (radial-circumferential)*direction[:, :, None]*direction[:, None, :]
    boundary = ids[[0, -1]].ravel()
    wall_values = radii[[0, -1]]**power
    model = thermal.ThermalModel(
        torch.from_numpy(nodes), torch.from_numpy(conn), torch.from_numpy(np.tile(tensors, (n, 1, 1))),
        1., 1., torch.from_numpy(boundary), torch.from_numpy((wall_values[:, None]*angular).ravel()),
        element_type="q4")
    fem = thermal.solve_steady_thermal(model, sparse=True).numpy().reshape(ids.shape)
    fv, rates = annulus.solve_annular_conduction(radii, radial, 1., wall_values, 0.,
        azimuthal_mode=mode, circumferential_conductivity=circumferential, distributed_source=True)
    # Interpolate FEM along radial edges to compare at identical physical points.
    fem_mid = .5*(fem[1:] + fem[:-1])
    fv_mid = fv[:, None]*angular
    exact = centers[:, None]**power*angular
    exact_rate = -2*np.pi*radial*power*radii**power
    return dict(radial_cells=n, sectors=len(angles),
                fem_error_K=float(np.max(abs(fem_mid-exact))),
                fvm_error_K=float(np.max(abs(fv_mid-exact))),
                solver_difference_K=float(np.max(abs(fem_mid-fv_mid))),
                fvm_modal_rate_error_W_per_m=float(np.max(abs(rates-exact_rate))))


if __name__ == "__main__":
    runs = [measure(n) for n in (8, 16, 32, 64)]
    for coarse, fine in zip(runs, runs[1:]):
        for key in ("fem_error_K", "fvm_error_K", "solver_difference_K", "fvm_modal_rate_error_W_per_m"):
            order = float(np.log2(coarse[key]/fine[key]))
            fine[key+"_order"] = order
            if fine["radial_cells"] >= 32 and order < 1.8:
                raise RuntimeError(f"{key} convergence order {order} on {fine['radial_cells']} cells")
    print(json.dumps(dict(
        reference="Steady source-free unit-depth annulus, T=r^p*cos(3*theta+0.37), p=3*sqrt(k_theta/k_r), k_r=2 and k_theta=0.7 W/(m K). Dirichlet walls at 0.4 and 1 m. Maximum temperature errors at shared radial-edge midpoints. FVM modal rate is 2*pi*r times outward radial flux amplitude.",
        reproduce="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFEM/src:../TensorFVM/src python examples/cross_orthotropic_annulus.py",
        benchmark=provenance(__file__), FEM=provenance(thermal.__file__), FVM=provenance(annulus.__file__),
        runs=runs), indent=2, allow_nan=False))
