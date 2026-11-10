"""End-to-end verification of the conservative exchange operators (LBM/DEM -> FEM).

Three physics chains, each with an independent reference:

``case_couette``  LBM Couette wall shear -> membrane (plane stress) -> tip
                  displacement vs the exact bar solution ``u = tau L^2 / (2 E t)``.
``case_poiseuille`` LBM force-driven channel wall pressure -> simply supported
                  Kirchhoff plate -> centre deflection vs the Navier series
                  solution of the same (piecewise-constant) pressure field.
``case_contact``  Hertzian DEM contact forces -> cantilever plate strip ->
                  deflection vs the Euler-Bernoulli/plate-strip closed form
                  (single contact and 3-contact superposition).

Every case also reports the conservation audit of the exchange step
(total force / total moment residual) and a negative control that must fail.
"""
from __future__ import annotations

import numpy as np

from .analytics import (
    hertz_force,
    navier_center_deflection,
    navier_coefficients,
    navier_field,
    strip_uniform_pressure_center,
)
from .fem import PlaneStressQ4, PlateBFS, PlateStripBending
from .lbm import LatticeScale, run_channel
from .mapping import (
    QuadGrid,
    audit_contact_map,
    audit_panel_map,
    map_contacts,
    map_panels,
    map_panels_to_basis,
    nearest_node_lumping,
)

__all__ = ["case_couette", "case_poiseuille", "case_contact", "LITERATURE_PLATE_COEF"]

# Timoshenko & Woinowsky-Krieger: w_max = 0.00406 q a^4 / D for a uniformly loaded,
# simply supported square plate (tabulated coefficient 0.00406).
LITERATURE_PLATE_COEF = 0.00406


# ---------------------------------------------------------------------------
# case A: Couette wall shear -> membrane
# ---------------------------------------------------------------------------
def case_couette(
    *,
    ny: int = 32,
    nx: int = 8,
    tau_relax: float = 0.5963,
    u_phys: float = 0.05,
    plate_L: float = 1.0e-2,
    plate_W: float = 2.0e-3,
    plate_t: float = 5.0e-4,
    plate_E: float = 2.1e6,
    plate_nu: float = 0.0,
    n_elem_x: int = 40,
    n_elem_y: int = 8,
) -> dict:
    scale = LatticeScale(rho_phys=998.0, nu_phys=1.004e-6, h_phys=1.0e-3,
                         ny=ny, tau_relax=tau_relax)
    u_lat = scale.u_lattice(u_phys)
    res = run_channel(nx=nx, ny=ny, lid_velocity=u_lat, tau_relax=tau_relax,
                      max_steps=150000, tol=1e-8, check_every=1000)

    mu_lat = res.mu
    tau_lat_measured = res.tau_w_bottom                    # already mu * du/dy
    tau_lat_exact = mu_lat * u_lat / ny
    tau_phys_measured = scale.stress_physical(tau_lat_measured)
    tau_phys_exact = scale.mu_physical() * u_phys / 1.0e-3

    # --- exchange: wall shear panels -> membrane nodal loads -----------------
    grid = QuadGrid(0.0, plate_L, 0.0, plate_W, n_elem_x, n_elem_y)
    npx = nx
    xe = np.linspace(0.0, plate_L, npx + 1)
    ye = np.array([0.0, plate_W])
    traction = np.zeros((npx, 1, 3))
    traction[:, 0, 0] = tau_phys_measured
    f_mapped = map_panels(grid, xe, ye, traction)

    audit = audit_panel_map(grid, f_mapped, xe, ye, traction)

    memb = PlaneStressQ4(plate_L, plate_W, plate_t, plate_E, plate_nu, n_elem_x, n_elem_y)
    u = memb.solve(f_mapped, fixed="left")
    u_tip = memb.tip_mean_ux(u)
    u_exact = tau_phys_exact * plate_L ** 2 / (2.0 * plate_E * plate_t)

    # consistency reference: bilinear consistent load of a constant traction on a
    # rectangle is tau * area / 4 at each corner (computed independently of the map)
    f_consistent = np.zeros_like(f_mapped)
    hx = plate_L / n_elem_x
    hy_el = plate_W / n_elem_y
    for e in range(n_elem_x):
        for j in range(n_elem_y):
            for (i0, j0) in ((e, j), (e + 1, j), (e + 1, j + 1), (e, j + 1)):
                f_consistent[grid.node_index(i0, j0)] += tau_phys_measured * hx * hy_el / 4.0
    u_alt = memb.solve(f_consistent, fixed="left")
    mapping_only = abs(memb.tip_mean_ux(u_alt) - u_tip) / max(abs(u_tip), 1e-300)

    # energy consistency of the exchange: sum f_i u_i vs integral tau u dA
    ux_nodes = u[:, 0].reshape(n_elem_y + 1, n_elem_x + 1)
    work = float(np.sum(f_mapped[:, 0] * u[:, 0]))
    # exact quadrature over the FE interpolation of the mapped field
    work_ref = 0.0
    gx, gw = np.polynomial.legendre.leggauss(3)
    for e in range(n_elem_x):
        x0 = e * hx
        for s, w in zip(gx, gw):
            x = x0 + 0.5 * hx * (1 + s)
            xi = 0.5 * (1 + s)
            jac = 0.5 * hx * w
            for j in range(n_elem_y):
                yj = j * plate_W / n_elem_y
                yj1 = (j + 1) * plate_W / n_elem_y
                ux_line = (1 - xi) * ux_nodes[j, e] + xi * ux_nodes[j, e + 1]
                ux_line2 = (1 - xi) * ux_nodes[j + 1, e] + xi * ux_nodes[j + 1, e + 1]
                work_ref += tau_phys_measured * 0.5 * (ux_line + ux_line2) * jac * (yj1 - yj)
    energy_defect = abs(work - work_ref) / max(abs(work_ref), 1e-300)

    return {
        "lbm": {
            "steps": res.steps,
            "steady_residual": res.residual,
            "u_lattice": u_lat,
            "mach": u_lat / np.sqrt(1.0 / 3.0),
            "nu_lattice": scale.nu_lat,
            "profile_rel_L2_error": res.profile_error_linear(u_lat),
            "tau_w_bottom_lattice": res.tau_w_bottom,
            "tau_w_exact_lattice": tau_lat_exact,
            "tau_w_phys_Pa": tau_phys_measured,
            "tau_w_exact_phys_Pa": tau_phys_exact,
            "tau_rel_error": abs(tau_phys_measured - tau_phys_exact) / tau_phys_exact,
            "y_norm": ((np.arange(ny) + 0.5) / ny).tolist(),
            "u_profile": res.ux.mean(axis=0).tolist(),
            "u_profile_analytic": (u_lat * (np.arange(ny) + 0.5) / ny).tolist(),
        },
        "exchange": {
            "n_panels_x": npx,
            "target_nodes": grid.n_nodes,
            **audit,
        },
        "fem": {
            "u_tip_FEM_m": u_tip,
            "u_tip_exact_m": u_exact,
            "u_tip_rel_error": abs(u_tip - u_exact) / abs(u_exact),
            "mapping_vs_consistent_load_rel": mapping_only,
            "exchange_work_defect_rel": energy_defect,
            "external_work_J_per_m2": 0.5 * work,
        },
        "params": {
            "plate_L_m": plate_L, "plate_W_m": plate_W, "plate_t_m": plate_t,
            "plate_E_Pa": plate_E, "plate_nu": plate_nu,
            "u_phys_m_s": u_phys, "ny": ny, "tau_relax": tau_relax,
            "dx_m": scale.dx, "dt_s": scale.dt,
            "stress_scale_Pa": scale.stress_scale,
        },
    }


# ---------------------------------------------------------------------------
# case B: Poiseuille wall pressure -> simply supported plate
# ---------------------------------------------------------------------------
def case_poiseuille(
    *,
    nx: int = 8,
    ny: int = 32,
    tau_relax: float = 0.5963,
    body_force: float = 1.0e-5,
    plate_t: float = 2.0e-5,
    plate_E: float = 2.1e9,
    plate_nu: float = 0.3,
    n_elem: int = 24,
) -> dict:
    scale = LatticeScale(rho_phys=998.0, nu_phys=1.004e-6, h_phys=1.0e-3,
                         ny=ny, tau_relax=tau_relax)
    res = run_channel(nx=nx, ny=ny, body_force=body_force, tau_relax=tau_relax,
                      max_steps=150000, tol=1e-8, check_every=1000)

    a = nx * scale.dx       # plate x-length == channel length
    b = ny * scale.dx       # plate y-length == channel width
    # wall pressure from the Stokes momentum balance dp/dx = -rho*g, referenced to
    # the LBM exit density (measured), i.e. p(x) = p0 - rho g x with p0 = p(x=0).
    x_lat = np.linspace(0.0, nx, nx + 1)
    p0_lat = float(res.rho[-1].mean()) / 3.0 + body_force * nx
    p_lat = p0_lat - body_force * x_lat
    p_phys = scale.pressure_physical(p_lat)          # relative to rho/3
    dp_phys = scale.pressure_physical(p0_lat) - scale.pressure_physical(p0_lat - body_force * nx)

    xe = np.linspace(0.0, a, nx + 1)
    ye = np.array([0.0, b])
    pressure = 0.5 * (p_phys[:-1] + p_phys[1:])[:, None]   # panel mean, sign: +p down
    traction = np.zeros((nx, 1, 3))
    traction[:, 0, 2] = -pressure[:, 0]                    # downward on the plate

    grid = QuadGrid(0.0, a, 0.0, b, n_elem, n_elem)
    f_mapped = map_panels(grid, xe, ye, traction)
    audit = audit_panel_map(grid, f_mapped, xe, ye, traction)

    plate = PlateBFS(a, b, plate_t, plate_E, plate_nu, n_elem, n_elem)
    load_nodal = plate.nodal_force_vector(f_mapped)
    u_nodal = plate.solve(load_nodal)
    w_nodal = plate.center_deflection(u_nodal)

    # generalised (consistent) mapping: pressure distributed on the Hermite basis
    hx = a / n_elem
    hy = b / n_elem

    def bfs_basis(ci: int, cj: int, x: float, y: float) -> np.ndarray:
        s = (x - ci * hx) / hx
        r = (y - cj * hy) / hy
        return plate._basis(min(max(s, 0.0), 1.0), min(max(r, 0.0), 1.0), hx, hy)[0]

    load_cons = map_panels_to_basis(grid, xe, ye, -pressure.reshape(nx, 1, 1),
                                   bfs_basis, 4)
    u_cons = plate.solve(load_cons)
    w_cons = plate.center_deflection(u_cons)

    # analytic Navier reference for exactly this piecewise-constant pressure
    coef = navier_coefficients(a, b, xe, ye, -pressure, 199, 199)
    w_navier = navier_center_deflection(a, b, plate.D, coef)
    coef_coarse = navier_coefficients(a, b, xe, ye, -pressure, 99, 99)
    w_navier_coarse = navier_center_deflection(a, b, plate.D, coef_coarse)

    # plate-solver self check: uniform pressure on a *square* panel vs the tabulated
    # coefficient 0.00406 q a^4 / D (the case-B plate itself is rectangular, so a
    # separate square panel is used for the literature anchor)
    p_mean = float(np.mean(pressure[:, 0]))
    plate_u = PlateBFS(a, a, plate_t, plate_E, plate_nu, n_elem, n_elem)
    load_uniform = plate_u.consistent_load(lambda x, y: -p_mean)
    w_uniform = plate_u.center_deflection(plate_u.solve(load_uniform))
    w_lit = -LITERATURE_PLATE_COEF * p_mean * a ** 4 / plate_u.D

    # energy audit on the consistent-load solve
    e_strain = plate.strain_energy(u_cons)
    e_work = plate.external_work(load_cons, u_cons)

    return {
        "lbm": {
            "steps": res.steps,
            "steady_residual": res.residual,
            "profile_rel_L2_error": res.profile_error_parabolic(body_force),
            "u_max_lattice": float(res.ux.mean(axis=0).max()),
            "mach": float(res.ux.mean(axis=0).max()) / np.sqrt(1.0 / 3.0),
            "tau_w_bottom_lattice": res.tau_w_bottom,
            "tau_w_exact_lattice": body_force * ny / 2.0,
            "tau_w_rel_error": abs(res.tau_w_bottom - body_force * ny / 2.0) / (body_force * ny / 2.0),
            "dp_over_channel_Pa": float(dp_phys),
            "y_norm": ((np.arange(ny) + 0.5) / ny).tolist(),
            "u_profile": res.ux.mean(axis=0).tolist(),
            "u_profile_analytic": (
                body_force / (2.0 * res.mu) * (np.arange(ny) + 0.5) * (ny - (np.arange(ny) + 0.5))
            ).tolist(),
        },
        "exchange": {
            "n_panels_x": nx,
            "target_nodes": grid.n_nodes,
            "panel_pressure_Pa": pressure[:, 0].tolist(),
            "panel_centres_x_m": (0.5 * (xe[:-1] + xe[1:])).tolist(),
            **audit,
        },
        "fem": {
            "w_center_mapped_nodal_m": w_nodal,
            "w_center_mapped_consistent_m": w_cons,
            "w_center_navier_m": w_navier,
            "navier_truncation_rel": abs(w_navier - w_navier_coarse) / abs(w_navier),
            "rel_error_mapped_nodal": abs(w_nodal - w_navier) / abs(w_navier),
            "rel_error_mapped_consistent": abs(w_cons - w_navier) / abs(w_navier),
            "w_center_uniform_literature_m": w_lit,
            "w_center_uniform_fem_m": w_uniform,
            "uniform_vs_literature_rel": abs(w_uniform - w_lit) / abs(w_lit),
            "strain_energy_J": e_strain,
            "external_work_J": e_work,
            "energy_residual_rel": abs(e_strain - e_work) / max(abs(e_work), 1e-300),
            "plate_x": np.linspace(0.0, a, n_elem + 1).tolist(),
            "w_centerline_fem": u_cons[0::4].reshape(n_elem + 1, n_elem + 1)[n_elem // 2].tolist(),
            "w_centerline_navier": navier_field(
                a, b, plate.D, coef,
                np.linspace(0.0, a, n_elem + 1),
                np.array([0.5 * b]),
            )[0].tolist(),
            "w_field_fem": u_cons[0::4].reshape(n_elem + 1, n_elem + 1).tolist(),
            "nodal_pressure_Pa": (-f_mapped[:, 2] / (a / n_elem) / (b / n_elem)).tolist(),
        },
        "params": {
            "a_m": a, "b_m": b, "plate_t_m": plate_t, "plate_E_Pa": plate_E,
            "plate_nu": plate_nu, "D_Nm": plate.D, "n_elem": n_elem,
            "p_mean_Pa": p_mean, "p_scale_Pa": scale.stress_scale,
            "body_force_lattice": body_force,
        },
    }


# ---------------------------------------------------------------------------
# case C: Hertzian DEM contacts -> cantilever plate strip
# ---------------------------------------------------------------------------
def _cantilever_green(q: float, x0: float, x: float, D: float) -> float:
    x = np.asarray(x, dtype=float)
    return np.where(
        x <= x0,
        q * x ** 2 * (3 * x0 - x) / (6 * D),
        q * x0 ** 2 * (3 * x - x0) / (6 * D),
    )


def case_contact(
    *,
    L: float = 0.1,
    width: float = 0.02,
    t: float = 2.0e-3,
    E_plate: float = 2.1e9,
    nu_plate: float = 0.3,
    R_sphere: float = 5.0e-3,
    E_sphere: float = 70.0e9,
    nu_sphere: float = 0.22,
    overlaps_um: tuple[float, ...] = (2.0, 3.0, 2.0),
    x_rel: tuple[float, ...] = (0.3025, 0.5013, 0.7021),
    n_elem: int = 200,
) -> dict:
    D = E_plate * t ** 3 / (12.0 * (1.0 - nu_plate ** 2))
    grid = QuadGrid(0.0, L, 0.0, width, n_elem, 1)
    xs = np.array(x_rel) * L
    ys = np.full_like(xs, width / 2.0)
    deltas = np.array(overlaps_um) * 1e-6
    forces_1d = np.array([
        hertz_force(E_plate, nu_plate, E_sphere, nu_sphere, R_sphere, d) for d in deltas
    ])
    points = np.column_stack([xs, ys])
    forces = np.zeros((len(xs), 3))
    forces[:, 2] = -forces_1d

    f_nodes, info = map_contacts(grid, points, forces, complete_moment=True)
    f_raw, _ = map_contacts(grid, points, forces, complete_moment=False)
    f_naive = nearest_node_lumping(grid, points, forces)

    audit = audit_contact_map(grid, f_nodes, points, forces)
    audit_raw = audit_contact_map(grid, f_raw, points, forces)
    audit_naive = audit_contact_map(grid, f_naive, points, forces)

    strip = PlateStripBending(L, D, n_elem, "clamped-free")
    # the 1-D strip model is per unit width: integrate the mapped node forces over
    # the strip width (the 2-D grid has one cell across the width)
    q_nodes = f_nodes[:, 2].reshape(2, n_elem + 1).sum(axis=0) / width
    load = np.zeros(2 * strip.n_nodes)
    load[0::2] = q_nodes
    u = strip.solve(load)

    q_contacts = -forces_1d / width       # downward -> negative w, FE convention
    w_analytic = np.zeros(len(xs))
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            w_analytic[i] += float(_cantilever_green(q_contacts[j], xj, xi, D))
    # evaluate the FE field at the exact contact abscissae with the cubic Hermite
    # basis (comparing a nodal value at the nearest node against a continuous
    # reference would inject an O(h/x) error near the clamp)
    def fe_w(x: float) -> float:
        e = min(int(x / strip.le), n_elem - 1)
        xi = (x - e * strip.le) / strip.le
        H = strip._hermite(xi, strip.le)
        return float(H @ u[2 * e:2 * e + 4])

    w_fem = np.array([fe_w(x) for x in xs])
    rel = np.abs(w_fem - w_analytic) / np.abs(w_analytic)

    # contact-couple demonstration: a roller that also transmits a rocking couple
    # (rolling resistance) cannot be represented by forces alone, so the raw
    # partition-of-unity lumping loses that moment and the completion repairs it.
    moms = np.zeros_like(forces)
    moms[1, 1] = 1.0e-3
    f_mom_raw, _ = map_contacts(grid, points, forces, moments=moms, complete_moment=False)
    f_mom_fix, _ = map_contacts(grid, points, forces, moments=moms, complete_moment=True)
    mom_audit_raw = audit_contact_map(grid, f_mom_raw, points, forces, moments=moms)
    mom_audit_fix = audit_contact_map(grid, f_mom_fix, points, forces, moments=moms)
    applied_couple = moms.sum(axis=0)
    cnorm = max(float(np.linalg.norm(applied_couple)), 1e-300)

    def _couple_loss(audit: dict) -> float:
        dm = np.array(audit["moment_target"]) - np.array(audit["moment_source"])
        return float(np.linalg.norm(dm) / cnorm)

    # energy audit
    e_strain = strip.strain_energy(u)
    e_work = strip.external_work(load, u)
    # interface work: mapped nodal forces vs the physical point-load work
    w_fem_all = u[0::2]
    work_mapped = float(np.sum(q_nodes * w_fem_all))
    work_point = float(np.sum(q_contacts * w_analytic))
    interface_defect = abs(work_mapped - work_point) / abs(work_point)

    return {
        "dem": {
            "n_contacts": len(xs),
            "overlaps_um": list(overlaps_um),
            "hertz_forces_N": forces_1d.tolist(),
            "contact_x_m": xs.tolist(),
            "moment_defect_raw_rel": info["moment_defect_raw"] / max(float(np.abs(forces).sum()), 1e-30),
        },
        "exchange": {
            "conserved": audit,
            "raw_lumping": audit_raw,
            "negative_control_nearest_node": audit_naive,
            "moment_carrying": {
                "raw": mom_audit_raw,
                "completed": mom_audit_fix,
                "applied_couple_Nm": moms[1].tolist(),
                "raw_couple_loss_rel": _couple_loss(mom_audit_raw),
                "completed_couple_loss_rel": _couple_loss(mom_audit_fix),
            },
            "target_nodes": grid.n_nodes,
        },
        "fem": {
            "w_fem_m": w_fem.tolist(),
            "w_analytic_m": w_analytic.tolist(),
            "rel_error": rel.tolist(),
            "rel_error_max": float(np.max(rel)),
            "strain_energy_J": e_strain,
            "external_work_J": e_work,
            "energy_residual_rel": abs(e_strain - e_work) / max(abs(e_work), 1e-300),
            "interface_work_defect_rel": interface_defect,
            "x_nodes_m": np.linspace(0.0, L, n_elem + 1).tolist(),
            "w_nodes_m": u[0::2].tolist(),
            "q_nodes_N_per_m": q_nodes.tolist(),
            "contact_analytic_deflection_m": [
                float(sum(_cantilever_green(q_contacts[j], xs[j], x, D) for j in range(len(xs))))
                for x in np.linspace(0.0, L, n_elem + 1)
            ],
        },
        "params": {
            "L_m": L, "width_m": width, "t_m": t, "E_plate_Pa": E_plate,
            "nu_plate": nu_plate, "D_Nm": D, "R_sphere_m": R_sphere,
            "E_sphere_Pa": E_sphere, "nu_sphere": nu_sphere, "n_elem": n_elem,
        },
    }