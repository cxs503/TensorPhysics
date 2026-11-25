"""Verify harmonic BEM heat conduction against two independent planar thermal waves."""
import argparse
import json
import shlex
import sys

import numpy as np

from conduction_provenance import provenance
from tensorbem import geometry, harmonic_conduction


def subregion_heat(result, center, directions, amplitudes, polynomial, gradient, hessian, storage, wave, loss):
    """Check a rotated rectangle using boundary flux and independent area integrals."""
    angle = .37
    axes = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    half = np.array([.27, .19])
    origin = center+np.array([.19, -.12])
    area = 4*np.prod(half)
    nodes, weights = np.polynomial.legendre.leggauss(10)
    z = wave*(directions @ axes.T)*half
    # The mean exponential on [-a,a] is sinh(lambda*a)/(lambda*a).
    centered_modes = amplitudes*np.exp(wave*((origin-center) @ directions.T-1))
    mean_modes = centered_modes*np.prod(np.sinh(z)/z, axis=1)
    second = np.diag(axes @ hessian @ axes.T)
    mean_polynomial = polynomial(origin)+np.dot(half**2, second)/6
    generated = area*(storage*mean_polynomial-result.conductivity*np.trace(hessian))
    generated -= area*loss*mean_modes.sum()
    exact_storage = area*storage*(mean_polynomial+mean_modes.sum())

    uv = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1)
    interior = origin+(uv*half) @ axes
    temperature = result.temperature(interior.reshape(-1, 2)).reshape(len(nodes), len(nodes))
    stored = area*storage*(weights @ temperature @ weights)/4
    numeric_generation = generated-loss*(stored-exact_storage)/storage
    edge_points, inward_normals, edge_weights, exact_heats = [], [], [], []
    for axis in (0, 1):
        tangent = 1-axis
        for side in (-1, 1):
            midpoint = origin+side*half[axis]*axes[axis]
            edge_points.append(midpoint+nodes[:, None]*half[tangent]*axes[tangent])
            inward_normals.append(-side*axes[axis])
            edge_weights.append(half[tangent]*weights)
            edge_modes = centered_modes*np.exp(side*z[:, axis])*np.sinh(z[:, tangent])/z[:, tangent]
            mean_gradient = gradient+midpoint @ hessian+wave*edge_modes @ directions
            exact_heats.append(2*half[tangent]*result.conductivity*side*(mean_gradient @ axes[axis]))
    flux = result.heat_flux(np.concatenate(edge_points)).reshape(4, len(nodes), 2)
    face_heat = np.einsum("eqd,ed,eq->e", flux, inward_normals, edge_weights)
    inward = face_heat.sum()
    return dict(subregion_face_error_W_m=float(abs(face_heat-exact_heats).max()),
                subregion_inward_error_W_m=float(abs(inward-(exact_storage-generated))),
                subregion_storage_error_W_m=float(abs(stored-exact_storage)),
                subregion_balance_W_m=float(abs(inward+numeric_generation-stored)),
                inward_W_m=[inward.real, inward.imag], generated_W_m=[numeric_generation.real, numeric_generation.imag],
                stored_W_m=[stored.real, stored.imag])


def measure(n, omega, condition="temperature", *, quadratic=False, graded=False, heated=False, affine=False, curved_source=False, interior_flux=False, subregion=False, curved_reservoir=False):
    curved_source = curved_source or curved_reservoir
    if graded:
        local = np.arange(n//4)/(n//4)
        angle = (np.arange(4)[:, None]+(1-np.cos(np.pi*local))/2).ravel()*np.pi/2
        mesh = geometry.PanelMesh(np.column_stack((np.cos(angle), np.sin(angle))))
    else:
        mesh = geometry.regular_polygon(n)
    # Offset the affine-source domain so its gradient contributes net power.
    center = np.array([.35, -.2]) if affine or curved_source else np.zeros(2)
    if affine or curved_source:
        mesh = geometry.PanelMesh(mesh.vertices+center)
    points = harmonic_conduction.harmonic_boundary_points(mesh, quadratic=quadratic)
    count = len(points)
    normals = np.repeat(mesh.normals, count//n, axis=0)
    k, capacity = 1.7, 2.3
    loss = 2.8 if curved_reservoir else 0.
    admittance = loss+1j*omega*capacity
    wave = np.sqrt(admittance/k)
    directions = np.array([[.6, .8], [-.8, .6]])
    amplitudes = np.array([.7-.2j, -.3+.4j])

    def modes(points):
        return amplitudes*np.exp(wave*((points-center) @ directions.T-1))

    boundary = modes(points)
    base = -.25+.4j if heated or affine or curved_source else 0j
    gradient = np.array([.31+.17j, -.23+.42j]) if affine or curved_source else np.zeros(2)
    source_gradient = admittance*gradient
    hessian = (np.array([[.34+.12j, -.28+.19j], [-.28+.19j, .16-.4j]])
               if curved_source else np.zeros((2, 2)))
    storage = 1j*omega*capacity
    source = admittance*base-k*np.trace(hessian)
    coefficients = admittance*np.array([hessian[0, 0]/2, hessian[0, 1], hessian[1, 1]/2])

    # Supply all curvature through the reservoir in the exchange case.
    reservoir = coefficients/loss if curved_reservoir else np.zeros(3)
    source_quadratic = np.zeros(3) if curved_reservoir else coefficients

    def polynomial(xy):
        return base+xy @ gradient+.5*np.sum((xy @ hessian)*xy, axis=-1)

    wall = polynomial(points)+boundary.sum(1)
    flux = k*(wave*np.sum(boundary*(normals @ directions.T), axis=1)
              +np.sum(normals*(gradient+points @ hessian), axis=1))
    # Fixed arcs stay in place as the polygon is refined.
    kinds = (np.repeat(["temperature", "flux", "film", "film"], count//4)
             if condition == "mixed" else np.full(count, condition))
    film = 2.5+.4*points[:, 0]
    values = np.select([kinds == "flux", kinds == "film"], [flux, wall+flux/film], default=wall)
    result = harmonic_conduction.solve_mixed_harmonic_conduction(
        mesh, kinds, values, omega, conductivity=k, volumetric_heat_capacity=capacity, film_coefficients=film, quadratic=quadratic,
        volumetric_source=source, source_gradient=source_gradient, source_quadratic=source_quadratic,
        heat_loss_coefficient=loss, bulk_ambient_quadratic=reservoir)
    angles = np.arange(21)*2*np.pi/21+.137
    probes = center+np.concatenate([r*np.column_stack((np.cos(angles), np.sin(angles))) for r in (.15, .55, .83)])
    extra = {}
    if subregion:
        extra.update(subregion_heat(result, center, directions, amplitudes, polynomial, gradient, hessian, storage, wave, loss))
    if interior_flux:
        analytic = -k*(gradient+probes @ hessian+wave*modes(probes) @ directions)
        error = abs(result.heat_flux(probes)-analytic)
        extra["interior_x_flux_error_W_m2"], extra["interior_y_flux_error_W_m2"] = error.max(axis=0).tolist()
    if heated or affine or curved_source:
        # Integrate the analytic storage over the actual polygon, using
        # triangles from the origin to each panel. No BEM flux enters storage.
        nodes, weights = np.polynomial.legendre.leggauss(10)
        u = (nodes+1)/2
        edge = mesh.vertices[:, None, :]+u[None, :, None]*mesh.vectors[:, None, :]
        xy = edge[:, :, None, :]*u[None, None, :, None]
        jac = mesh.vertices[:, 0]*mesh.ends[:, 1]-mesh.vertices[:, 1]*mesh.ends[:, 0]
        field = polynomial(xy)+modes(xy.reshape(-1, 2)).sum(1).reshape(n, len(u), len(u))
        volume_temperature = np.einsum("p,i,j,pij->", jac, weights/2, weights*u/2, field)
        panel_weights = np.array([5., 8., 5.])/18 if quadratic else np.array([1.])
        inward_rate = mesh.lengths @ (result.inward_flux.reshape(n, -1) @ panel_weights)
        generation = storage*polynomial(xy)-k*np.trace(hessian)
        if curved_reservoir:
            x, y = xy[..., 0], xy[..., 1]
            ambient = reservoir[0]*x*x+reservoir[1]*x*y+reservoir[2]*y*y
            generation = source+xy @ source_gradient+loss*(ambient-field)
        generated = np.einsum("p,i,j,pij->", jac, weights/2, weights*u/2, generation)
        extra["energy_residual_W_m"] = float(abs(inward_rate+generated-1j*omega*capacity*volume_temperature))
    worst = np.argmax(abs(result.inward_flux-flux))
    return dict(**extra, quadratic=quadratic, graded=graded, boundary_unknowns=count, worst_flux_panel=int(worst//(count//n)), worst_flux_condition=str(kinds[worst]), panels=n, omega_rad_s=omega, boundary=condition,
                wall_temperature_error_K=float(abs(result.wall_temperature-wall).max()), interior_error_K=float(abs(result.temperature(probes)-(polynomial(probes)+modes(probes).sum(1))).max()),
                boundary_flux_error_W_m2=float(abs(result.inward_flux-flux).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("temperature", "flux", "film", "mixed"), default="temperature")
    parser.add_argument("--quadratic", action="store_true", help="use discontinuous quadratic panel traces")
    parser.add_argument("--graded", action="store_true", help="cluster panel ends at quadrant boundaries")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--affine", action="store_true", help="verify an affine volume source on a translated polygon")
    source_group.add_argument("--curved-source", action="store_true", help="verify quadratic volumetric heating with a cross term")
    source_group.add_argument("--heated", action="store_true", help="add uniform volumetric generation and check total heat balance")
    source_group.add_argument("--curved-reservoir", action="store_true", help="drive x², xy and y² curvature through volumetric exchange")
    parser.add_argument("--interior-flux", action="store_true", help="verify both Cartesian interior heat-flux components")
    parser.add_argument("--subregion", action="store_true", help="compare integrated flux and storage inside a rotated rectangle")
    args = parser.parse_args()
    rows = []
    for omega in (1., 6.):
        previous = None
        refinements = (16, 32, 64, 128, 256) if args.boundary == "mixed" else (16, 32, 64, 128)
        for n in refinements:
            row = measure(n, omega, args.boundary, quadratic=args.quadratic, graded=args.graded, heated=args.heated, affine=args.affine,
                          curved_source=args.curved_source, curved_reservoir=args.curved_reservoir, interior_flux=args.interior_flux, subregion=args.subregion)
            metrics = ["interior_error_K"]
            if args.boundary != "flux":
                metrics.append("boundary_flux_error_W_m2")
            if args.boundary != "temperature":
                metrics.append("wall_temperature_error_K")
            if args.interior_flux:
                metrics.extend(["interior_x_flux_error_W_m2", "interior_y_flux_error_W_m2"])
            if args.subregion:
                metrics.extend(["subregion_face_error_W_m", "subregion_inward_error_W_m", "subregion_storage_error_W_m"])
            errors = np.array([row[key] for key in metrics])
            row["order_metrics"] = metrics
            orders = np.log2(previous/errors).tolist() if previous is not None else []
            row.update(orders=orders, passed=bool(np.isfinite(errors).all() and all(p > 0 for p in orders)
                       and row.get("subregion_balance_W_m", 0.) < 1e-9
                       and (n < 128 or (min(orders) > 1.8 and row.get("energy_residual_W_m", 0.) < .005))))
            rows.append(row)
            previous = errors
    command = shlex.join(["python", "examples/bem_harmonic_conduction.py", *sys.argv[1:]])
    print(json.dumps(dict(reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorBEM/src "+command, runs=rows, sources=[provenance(p) for p in
        (__file__, harmonic_conduction.__file__, geometry.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
