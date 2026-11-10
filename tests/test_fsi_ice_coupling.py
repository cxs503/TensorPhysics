"""Verification and coupling tests for the 3-D fluid-structure-ice prototype.

Every solver is checked against an *analytic* reference before any coupled
result is trusted:

* LBM  -- Poiseuille profile, mass conservation, zero spurious force at rest,
          free-surface hydrostatics.
* DEM  -- Hertz contact force, coefficient of restitution, Stokes terminal
          velocity.
* FEM  -- uniaxial patch test, cantilever beam deflection, conservative load
          mapping, face traction resultant.

The coupling tests then check the contracts (mass/momentum hand-off), the
sub-cycling clock, single-field degradation (a coupled run with one field
switched off must reproduce the standalone solver bit for bit) and the
`RunManifest` provenance record.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.fsi_ice.contracts import (
    CONTRACT_VERSION,
    FieldExchangeError,
    FluidFieldPacket,
    ParticlePacket,
    StructurePacket,
    check_energy_handoff,
    check_mass_conservation,
    check_momentum_balance,
)
from tensorphysics.fsi_ice.dem import (
    DEMConfig,
    DEMWorld,
    HertzMindlinContact,
    IceSheetFracture,
    WedgeBow,
)
from tensorphysics.fsi_ice.fem import ElastostaticHull, HullFEMConfig
from tensorphysics.fsi_ice.lattice import (
    FreeSurfaceConfig,
    LatticeBoltzmannSolver,
    LatticeSpec,
)
from tensorphysics.fsi_ice.runner import CouplingConfig, CouplingRunner


# ---------------------------------------------------------------------------
# fluid kernel
# ---------------------------------------------------------------------------
def test_poiseuille_profile_matches_analytic():
    """Body-force driven 2-D channel flow vs the exact parabolic solution."""
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    nz, force = 12, 1e-5
    spec = LatticeSpec(nx=4, ny=1, nz=nz, dim=2, tau=tau, gravity=(force, 0, 0),
                       x_periodic=True, initial_velocity=(0, 0, 0), top_boundary="wall")
    solver = LatticeBoltzmannSolver(spec)
    for _ in range(6000):
        solver.step()
    u = solver.u[:, 0, 2, 0]
    half = nz / 2.0  # wall planes sit half a cell outside the first/last cell
    zz = np.arange(nz) - (nz - 1) / 2.0
    analytic = force / (2.0 * nu) * (half**2 - zz**2)
    err = np.abs(u - analytic).max() / analytic.max()
    assert err < 0.03, f"Poiseuille error {err:.3%} exceeds 3%"


def test_lattice_mass_is_conserved():
    spec = LatticeSpec(nx=8, ny=1, nz=8, dim=2, tau=0.7, gravity=(1e-5, 0, 0),
                       x_periodic=True, initial_velocity=(0, 0, 0), top_boundary="wall")
    solver = LatticeBoltzmannSolver(spec)
    m0 = solver.f.sum()
    for _ in range(200):
        solver.step()
    assert abs(solver.f.sum() - m0) / m0 < 1e-12


def test_fluid_at_rest_exerts_no_dynamic_force_on_body():
    """Static momentum flux is subtracted, so a body in still water is force free."""
    spec = LatticeSpec(nx=20, ny=8, nz=10, dim=3, tau=0.7, inlet_velocity=(0.0, 0.0, 0.0))
    solver = LatticeBoltzmannSolver(spec)
    mask = np.zeros(spec.shape, dtype=bool)
    mask[3:7, 3:5, 8:12] = True
    solver.set_solid(mask)
    for _ in range(50):
        force = solver.step()
    assert np.all(np.isfinite(force))
    assert np.linalg.norm(force) < 1e-6, f"spurious force {force}"


def test_free_surface_hydrostatic_pressure():
    """p = rho g h at the tank bottom for a mass-tracking free surface."""
    g, depth = 1e-4, 6
    spec = LatticeSpec(nx=6, ny=1, nz=12, dim=2, tau=0.9, gravity=(0, 0, -g),
                       initial_velocity=(0, 0, 0), top_boundary="slip")
    solver = LatticeBoltzmannSolver(spec, FreeSurfaceConfig(enabled=True, initial_depth=depth))
    m0 = solver.total_mass
    for _ in range(1200):
        solver.step()
    expected_rho = 1.0 + 3.0 * g * depth
    assert abs(solver.rho[0].mean() - expected_rho) / expected_rho < 0.01
    assert abs(solver.total_mass - m0) / m0 < 1e-3


# ---------------------------------------------------------------------------
# DEM kernel
# ---------------------------------------------------------------------------
def test_hertz_contact_force_matches_analytic():
    contact = HertzMindlinContact(400.0, 0.3)
    radius, overlap = 1.0, 0.01
    expected = 4.0 / 3.0 * contact.e_star * np.sqrt(radius) * overlap**1.5
    assert contact.normal_force(radius, overlap) == pytest.approx(expected, rel=1e-12)


def test_restitution_coefficient_is_recovered():
    """Tsuji damping reproduces the target restitution for e = 0.9."""
    target = 0.9
    w = DEMWorld(
        DEMConfig(youngs_modulus=400.0, restitution=target, gravity=(0, 0, 0),
                  drag_enabled=False, dim=3, friction=0.0),
        [[-50, -50, -50], [50, 50, 50]],
    )
    w.add_particles([[0.8, 0, 0], [2.2, 0, 0]], [0.5, 0.5])
    w.velocity[0] = [0.01, 0, 0]
    w.velocity[1] = [-0.01, 0, 0]
    for _ in range(40000):
        w.step(0.02)
        if w.velocity[0, 0] < 0:
            break
    for _ in range(40000):
        w.step(0.02)
        if w.velocity[0, 0] > 0.0002:
            break
    measured = abs(w.velocity[0, 0]) / 0.01
    # the linear (Tsuji) damper is capped by the spring force, which biases the
    # recovered restitution slightly high; 6% is the documented model accuracy
    assert measured == pytest.approx(target, abs=0.06), f"restitution {measured:.3f}"


def test_stokes_terminal_velocity():
    """Single particle in still fluid: v = m g / (3 pi mu d) for Re << 1."""
    cfg = DEMConfig(youngs_modulus=400.0, gravity=(0, 0, -1e-4), rho_particle=0.9,
                    rho_fluid=1.0, fluid_viscosity=4.5e-3, restitution=0.2, dim=3)
    w = DEMWorld(cfg, [[0, 0, -100], [10, 10, 100]])
    w.add_particles([[5, 5, 5]], [0.5])
    for _ in range(20000):
        w.step(0.05, fluid_velocity=np.zeros((1, 3)), water_level=None)
    mass = w.mass[0]
    radius = w.radius[0]
    analytic = -mass * 1e-4 / (3.0 * np.pi * 4.5e-3 * 2.0 * radius)
    assert w.velocity[0, 2] == pytest.approx(analytic, rel=0.02)


def test_dem_buoyancy_keeps_ice_afloat():
    """Coarse-grained particles use the ice volume for buoyancy, not the sphere volume."""
    cfg = DEMConfig(youngs_modulus=90.0, gravity=(0, 0, -0.049), rho_particle=0.9,
                    rho_fluid=1.0, drag_enabled=False, dim=3)
    w = DEMWorld(cfg, [[0, 0, 0], [20, 20, 20]])
    w.add_particles([[10, 10, 10]], [0.9], mass=np.array([7.2]))
    for _ in range(400):
        w.step(0.05, water_level=15.0)
    assert w.velocity[0, 2] > 0.0, "sea ice must float, not sink"


# ---------------------------------------------------------------------------
# FEM kernel
# ---------------------------------------------------------------------------
def _uniaxial_hull():
    cfg = HullFEMConfig(nx=6, ny=2, nz=2, size=(6.0, 2.0, 2.0), youngs_modulus=100.0,
                        poisson=0.25, density=1.0, clamp_face="xmin_axial")
    return ElastostaticHull(cfg)


def test_uniaxial_patch_test_is_exact():
    hull = _uniaxial_hull()
    load = hull.face_loads(0, True, [1.0, 0.0, 0.0])
    assert np.allclose(load.reshape(-1, 3).sum(axis=0), [4.0, 0.0, 0.0], atol=1e-12)
    u = hull.static_solve(load).reshape(-1, 3)
    analytic = hull.mesh.nodes[:, 0] / hull.cfg.youngs_modulus
    assert np.abs(u[:, 0] - analytic).max() < 1e-10


def test_face_traction_resultant_equals_traction_times_area():
    hull = _uniaxial_hull()
    for axis, area in ((0, 4.0), (1, 12.0), (2, 12.0)):
        load = hull.face_loads(axis, True, [0.5, 0.0, 0.0]).reshape(-1, 3).sum(axis=0)
        assert load[0] == pytest.approx(0.5 * area, rel=1e-12)


def test_cantilever_matches_euler_bernoulli():
    cfg = HullFEMConfig(nx=24, ny=4, nz=4, size=(10.0, 2.0, 2.0), youngs_modulus=100.0,
                        poisson=0.3, density=1.0, clamp_face="xmin")
    hull = ElastostaticHull(cfg)
    tip_load = 1e-3
    f = hull.face_loads(0, True, [0.0, 0.0, -tip_load / 4.0])
    u = hull.static_solve(f).reshape(-1, 3)
    tip = np.isclose(hull.mesh.nodes[:, 0], 10.0)
    numeric = u[tip, 2].mean()
    E, L, b, h = 100.0, 10.0, 2.0, 2.0
    analytic = -tip_load * L**3 / (3.0 * E * (b * h**3 / 12.0))
    assert numeric == pytest.approx(analytic, rel=0.05), f"{numeric} vs {analytic}"


def test_load_mapping_is_conservative():
    hull = _uniaxial_hull()
    points = np.array([[1.0, 0.5, 0.5], [4.0, 1.5, 1.5], [5.9, 1.0, 1.0]])
    forces = np.array([[0.3, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, -0.1]])
    nodal = hull.map_contact_loads(points, forces).reshape(-1, 3)
    assert np.allclose(nodal.sum(axis=0), forces.sum(axis=0), atol=1e-12)


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------
def test_contract_packets_validate():
    pos = np.zeros((2, 3))
    pkt = ParticlePacket(time=0.0, position=pos, velocity=pos.copy(),
                         radius=np.ones(2), mass=np.ones(2))
    pkt.validate()
    assert pkt.schema == CONTRACT_VERSION
    with pytest.raises(FieldExchangeError):
        ParticlePacket(time=0.0, position=pos, velocity=pos.copy(),
                       radius=-np.ones(2), mass=np.ones(2)).validate()


def test_mass_handoff_gate_rejects_lost_mass():
    pos = np.zeros((2, 3))
    pkt = ParticlePacket(time=0.0, position=pos, velocity=pos.copy(),
                         radius=np.ones(2), mass=np.array([0.9, 0.9]))
    err = check_mass_conservation({"volume": np.array([1.0, 1.0]), "density": 0.9}, pkt)
    assert err < 1e-12
    bad = ParticlePacket(time=0.0, position=pos, velocity=pos.copy(),
                         radius=np.ones(2), mass=np.array([0.9, 0.45]))
    with pytest.raises(FieldExchangeError):
        check_mass_conservation({"volume": np.array([1.0, 1.0]), "density": 0.9}, bad)


def test_momentum_handoff_gate():
    pos = np.zeros((2, 3))
    loads = np.array([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    pkt = StructurePacket(time=0.0, node_positions=pos, node_velocities=pos.copy(), loads=loads)
    assert check_momentum_balance(pkt, [0.5, 0.0, 0.0]) < 1e-12
    with pytest.raises(FieldExchangeError):
        check_momentum_balance(pkt, [0.9, 0.0, 0.0])


def test_energy_handoff_gate():
    assert check_energy_handoff([1.0, 2.0], [1.0, 2.0]) < 1e-12
    with pytest.raises(FieldExchangeError):
        check_energy_handoff([1.0, 2.0], [1.0, 1.0])


# ---------------------------------------------------------------------------
# fracture / fragmentation model
# ---------------------------------------------------------------------------
def test_plate_characteristic_length_and_break_force():
    frac = IceSheetFracture(thickness=2.0, flexural_strength=7.0, youngs_modulus=9e4,
                            poisson=0.3, rho_fluid=1.0, gravity=0.049, width=20.0,
                            cell_length=2.0, dim=3)
    D = 9e4 * 8.0 / (12.0 * (1 - 0.09))
    assert frac.L_b == pytest.approx((D / 0.049) ** 0.25, rel=1e-12)
    assert frac.q_c == pytest.approx(4.0 * frac.m_u / frac.L_b, rel=1e-12)
    assert frac.break_force == pytest.approx(frac.q_c * frac.cell_width, rel=1e-12)


def test_3d_fragment_is_smaller_than_2d_slab():
    common = dict(thickness=2.0, flexural_strength=7.0, youngs_modulus=9e4, poisson=0.3,
                  rho_fluid=1.0, gravity=0.049, width=20.0, cell_length=2.0)
    f3 = IceSheetFracture(dim=3, lateral_crack_spacing=2.0, **common)
    f2 = IceSheetFracture(dim=2, lateral_crack_spacing=2.0, **common)
    assert f2.n_lateral == 1 and f3.n_lateral > 1
    assert f3.equivalent_diameter() < f2.equivalent_diameter()


# ---------------------------------------------------------------------------
# coupling
# ---------------------------------------------------------------------------
def _mini_config(**kw):
    base = dict(dim=3, ny=12, nx=32, nz=12, water_level=9, x_bow_tip=20.0,
                x_bow_transom=28.0, keel_z=3.0, ice_width=8.0, n_steps=6,
                fluid_steps_per_couple=1, dem_dt=0.1)
    base.update(kw)
    return CouplingConfig(**base)


def test_coupled_run_manifest_is_valid():
    res = CouplingRunner(_mini_config(), label="mini").run()
    manifest = res.manifest
    manifest.verify_config_digest()
    data = manifest.to_dict()
    assert data["run_id"].startswith("fsi-ice-mini")
    assert data["config"]["dim"] == 3
    assert data["validation_status"] in ("passed", "failed")


def test_coupled_run_is_finite():
    res = CouplingRunner(_mini_config(), label="mini").run()
    assert np.all(np.isfinite(res.force_history))
    assert not res.diagnostics["diverged"]


def test_fluid_only_degradation_matches_standalone_solver():
    """With every other field switched off the coupled run is the LBM run."""
    cfg = _mini_config(coupling_mode="fluid_only", n_steps=4)
    res = CouplingRunner(cfg, label="mini").run()
    spec = cfg.lattice_spec()
    ref = LatticeBoltzmannSolver(spec)
    hull = WedgeBow(x_tip=cfg.x_bow_tip, half_angle_deg=cfg.bow_half_angle_deg,
                    z_bottom=cfg.keel_z, z_top=float(cfg.water_level),
                    x_max=cfg.x_bow_transom, y_center=cfg.ny / 2.0)
    ref.set_solid(hull.mask(spec.shape))
    for _ in range(cfg.n_steps * cfg.fluid_steps_per_couple):
        ref.step()
    runner = CouplingRunner(cfg, label="mini")
    runner.run()
    assert np.allclose(runner.fluid.f, ref.f, atol=1e-14)


def test_fem_only_degradation_stays_at_rest():
    res = CouplingRunner(_mini_config(coupling_mode="fem_only", n_steps=4), label="mini").run()
    assert res.diagnostics["max_fem_displacement"] == 0.0


def test_subcycling_clock_is_monotonic():
    cfg = _mini_config(n_steps=5, fluid_steps_per_couple=2)
    res = CouplingRunner(cfg, label="mini").run()
    assert len(res.times) == 5
    assert np.all(np.diff(res.times) > 0)
    assert res.times[-1] == pytest.approx(5 * 2.0)
    assert res.diagnostics["dem_substeps"] == max(1, int(round(2.0 / cfg.dem_dt)))


def test_energy_ledger_is_reported_and_bounded():
    res = CouplingRunner(_mini_config(n_steps=8), label="mini").run()
    ledger = res.ledger
    assert set(ledger) >= {"E_final", "E_injected", "residual", "relative_error"}
    assert np.isfinite(ledger["relative_error"])
    assert "fluid_energy_excluded" in ledger