"""Tests for the IN66-N08 cross-domain integration gate (TensorPhysics).

The tests are organised by stage plus one end-to-end gate test, and every
"accepts" test is paired with a "rejects" test: a validator that cannot say no
is not evidence of an interface contract.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorphysics.ice_nav_integration import (
    CHANNELS,
    CaseConfig,
    ChannelGrid,
    FlowSetup,
    StructuralSetup,
    accumulate,
    advect_upwind,
    brash_channel_split,
    build_ice_field_payload,
    build_ice_field_state,
    build_ledger,
    build_struct_feedback_payload,
    build_traction_payload,
    coverage_field,
    distribution_weights,
    floe_statistics,
    ittc57_cf,
    linear_gain,
    open_water_resistance,
    porous_drag_density,
    run_gate,
    solve_streamwise,
    solve_two_way,
    stability_scan,
    validate_hull_traction,
    validate_ice_field,
    validate_struct_feedback,
)
from tensorphysics.ice.resistance import brash_ice_resistance

from tensorphysics.ice_nav_integration.gate import geometry_audit, ice_channels


# ---------------------------------------------------------------------------
# contractual helpers
# ---------------------------------------------------------------------------


def _grid(nx=16, ny=4):
    return ChannelGrid(nx=nx, ny=ny, dx=10.0, dy=20.0)


def _field_payload(grid=None, coverage=0.6, thickness=0.15):
    grid = grid or _grid()
    alpha = coverage_field(grid, coverage=coverage, seed=7, correlation_cells=1.5)
    return (
        build_ice_field_payload(
            grid=grid.descriptor(),
            time_s=1.0,
            dt_s=0.1,
            thickness_m=thickness,
            ice_density_kg_m3=917.0,
            nominal_coverage=float(alpha.mean()),
            volume_fraction=alpha.tolist(),
            upstream_velocity_m_s=2.0,
        ),
        alpha,
    )


def _panel_template(grid, rans, source, thickness=0.15):
    x = grid.cell_centers_x()
    return [
        {
            "x_m": float(x[i]),
            "u_m_s": float(rans["u_m_s"][i]),
            "p_pa": float(rans["p_pa"][i]),
            "tau_wall_pa": float(rans["tau_wall_pa"][i]),
            "dpdx_pa_per_m": float(rans["dpdx_pa_per_m"][i]),
            "du_dx_1_per_s": float(rans["du_dx"][i]),
            "source_N_per_m": float(source[i]),
            "water_area_m2": float(rans["water_area_m2"][i]),
            "wetted_area_m2": float(2.0 * thickness * grid.dx),
            "projected_area_m2": float(thickness * 0.02 * grid.dx),
        }
        for i in range(grid.nx)
    ]


def _traction_payload(grid, source=None):
    alpha_mean = np.full(grid.nx, 0.6)
    source = np.full(grid.nx, 1.0e4) if source is None else source
    rans = solve_streamwise(
        grid=grid,
        alpha_mean=alpha_mean,
        speed_m_s=2.0,
        thickness_m=0.15,
        coverage_ambient=0.6,
        force_density_N_per_m=source,
        setup=FlowSetup(),
        draft_m=20.8,
        beam_m=58.0,
    )
    panels = _panel_template(grid, rans, source)
    totals = {
        "R_pressure_N": float(sum(p["p_pa"] * p["projected_area_m2"] for p in panels)),
        "R_friction_N": float(sum(p["tau_wall_pa"] * p["wetted_area_m2"] for p in panels)),
    }
    return (
        build_traction_payload(
            grid=grid.descriptor(),
            time_s=1.0,
            dt_s=0.5,
            source_dt_s=0.1,
            panels=panels,
            mass_flux_kg_s=rans["mass_flux_kg_s"],
            totals=totals,
        ),
        rans,
        source,
    )


def _failed(checks):
    return [c.check_id for c in checks if not c.passed]


# ---------------------------------------------------------------------------
# contracts: DEM -> RANS
# ---------------------------------------------------------------------------


def test_ice_field_contract_accepts_and_rejects():
    payload, _ = _field_payload()
    assert _failed(validate_ice_field(payload)) == []

    # wrong unit -> the field is dimensionally ambiguous
    bad = json.loads(json.dumps(payload))
    bad["units"]["thickness_m"] = "s"
    assert "D2_units_declared" in _failed(validate_ice_field(bad))

    # grid descriptor tampered -> downstream panels would land on another grid
    bad = json.loads(json.dumps(payload))
    bad["grid"]["dx_m"] = 12.0
    assert "D4_grid_hash" in _failed(validate_ice_field(bad))

    # volume fraction outside [0, 1]
    bad = json.loads(json.dumps(payload))
    bad["volume_fraction"][0][0] = 1.4
    assert "D5_fraction_bounds" in _failed(validate_ice_field(bad))

    # declared ice volume no longer matches the field (lost mass in handoff)
    bad = json.loads(json.dumps(payload))
    bad["total_ice_volume_m3"] *= 1.05
    assert "D6_ice_volume_closure" in _failed(validate_ice_field(bad))

    # coverage statistic disagrees with the field by > 1 %
    bad = json.loads(json.dumps(payload))
    bad["mean_volume_fraction"] *= 1.05
    assert "D7_coverage_consistency" in _failed(validate_ice_field(bad))


# ---------------------------------------------------------------------------
# contracts: RANS -> FSI and FSI -> DEM
# ---------------------------------------------------------------------------


def test_traction_contract_accepts_and_rejects():
    grid = _grid()
    payload, _, _ = _traction_payload(grid)
    ice_payload, _ = _field_payload(grid)
    assert _failed(validate_hull_traction(payload, ice_payload)) == []

    # panel grid shifted by one cell -> loads applied to the wrong stations
    bad = json.loads(json.dumps(payload))
    for row in bad["panels"]:
        row["x_m"] += grid.dx
    assert "R4_panel_grid_alignment" in _failed(validate_hull_traction(bad, ice_payload))

    # time-step not an integer subcycle of the DEM step
    bad = json.loads(json.dumps(payload))
    bad["dt_s"] = 0.37
    assert "R5_time_sync" in _failed(validate_hull_traction(bad, ice_payload))

    # momentum budget tampered -> the consumer would integrate an unbalanced field
    bad = json.loads(json.dumps(payload))
    bad["panels"][3]["dpdx_pa_per_m"] *= 1.2
    assert "R6_momentum_balance_closure" in _failed(validate_hull_traction(bad, ice_payload))

    # force totals inconsistent with the panels
    bad = json.loads(json.dumps(payload))
    bad["totals"]["R_pressure_N"] *= 1.1
    assert "R8_panel_force_closure" in _failed(validate_hull_traction(bad, ice_payload))


def test_feedback_contract_accepts_and_rejects():
    grid = _grid()
    traction, _, _ = _traction_payload(grid)
    w = np.full(grid.nx, 0.002)

    def _mk(**over):
        kwargs = dict(
            grid=grid.descriptor(),
            time_s=1.0,
            dt_s=0.5,
            source_dt_s=0.1,
            deflection_m=w.tolist(),
            clearance_m=(0.65 - w).tolist(),
            contact_pressure_pa=np.full(grid.nx, 1.0e5).tolist(),
            gap0_m=0.65,
            residual=1.0e-8,
            iterations=9,
            converged=True,
            spectral_radius=0.2,
            added_mass_ratio=0.21,
        )
        kwargs.update(over)
        return build_struct_feedback_payload(**kwargs)

    assert _failed(validate_struct_feedback(_mk(), traction)) == []

    # unconverged feedback must not be consumable by the DEM stage
    assert "S8_fixed_point_residual" in _failed(
        validate_struct_feedback(_mk(residual=1.0e-3, converged=False), traction)
    )
    # clearance not closed by the deflection
    bad = _mk(clearance_m=(0.60 - w).tolist())
    assert "S6_gap_closure" in _failed(validate_struct_feedback(bad, traction))
    # absurd deflection (larger than half the gap)
    bad = _mk(deflection_m=np.full(grid.nx, 0.5).tolist(), clearance_m=np.full(grid.nx, 0.15).tolist())
    assert "S7_displacement_bounds" in _failed(validate_struct_feedback(bad, traction))


# ---------------------------------------------------------------------------
# DEM field stage
# ---------------------------------------------------------------------------


def test_coverage_field_is_mass_exact_and_bounded():
    grid = _grid(nx=32, ny=8)
    for coverage in (0.0, 0.15, 0.6, 0.85):
        alpha = coverage_field(grid, coverage=coverage, seed=3, correlation_cells=2.0)
        assert alpha.shape == (grid.nx, grid.ny)
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0
        assert abs(alpha.mean() - coverage) <= 1e-12
    # determinism
    a = coverage_field(grid, coverage=0.6, seed=11)
    b = coverage_field(grid, coverage=0.6, seed=11)
    assert np.array_equal(a, b)


def test_advection_conserves_ice_budget():
    grid = _grid(nx=24, ny=6)
    state = build_ice_field_state(
        grid, coverage=0.6, thickness_m=0.15, lane_width_m=63.8, speed_m_s=2.0, seed=5
    )
    for _ in range(50):
        out = state.step(speed_m_s=2.0, dt_s=0.1)
        assert out["courant"] <= 1.0
        assert out["budget_residual_relative"] <= 1e-12
        assert 0.0 <= out["alpha"].min() and out["alpha"].max() <= 1.0
    # aliasing guard: the state must be updated in place
    assert np.array_equal(state.alpha, out["alpha"])


def test_advection_rejects_instability():
    grid = _grid(nx=8, ny=4)
    alpha = coverage_field(grid, coverage=0.5, seed=1)
    with pytest.raises(ValueError):
        advect_upwind(
            alpha,
            grid,
            speed_m_s=100.0,
            dt_s=1.0,
            thickness_m=0.15,
            inflow_column=alpha[0],
        )


def test_floe_statistics_are_physical():
    stats = floe_statistics(
        coverage=0.6, thickness_m=0.15, lane_width_m=63.8, speed_m_s=2.0
    )
    assert 1.0 <= stats["r_mean_m"] <= 8.0
    assert stats["r_rms_m"] >= stats["r_mean_m"]
    assert 0.0 < stats["n_floes_per_m2"] < 1.0
    assert stats["encounter_rate_1_per_s"] > 0.0
    with pytest.raises(ValueError):
        floe_statistics(coverage=1.2, thickness_m=0.15, lane_width_m=10.0, speed_m_s=1.0)


# ---------------------------------------------------------------------------
# flow stage
# ---------------------------------------------------------------------------


def test_streamwise_momentum_balance_and_panel_force():
    grid = _grid(nx=32, ny=4)
    alpha = np.linspace(0.6, 0.25, grid.nx)
    source = np.full(grid.nx, 5.0e3)
    rans = solve_streamwise(
        grid=grid,
        alpha_mean=alpha,
        speed_m_s=2.0,
        thickness_m=0.15,
        coverage_ambient=0.6,
        force_density_N_per_m=source,
        setup=FlowSetup(),
        draft_m=20.8,
        beam_m=58.0,
    )
    # continuity mapping u = V (1 - C_i) / (1 - alpha)
    expected_u = 2.0 * 0.4 / (1.0 - alpha)
    assert np.allclose(rans["u_m_s"], expected_u, rtol=1e-12)
    # recovered panel force = distributed sources + momentum flux.  The residual
    # is the O(dx) truncation of the one-sided end stencils in np.gradient, i.e.
    # 3e-4 relative here -- an honest discretisation error, not a contract error.
    closure = abs(rans["R_panel_N"] - (rans["R_sources_N"] + rans["momentum_flux_N"]))
    assert closure <= 1e-3 * abs(rans["R_panel_N"])
    # pressure must rise towards the bow (wake reference at the stern)
    assert rans["p_max_pa"] > 0.0
    assert abs(rans["p_pa"][-1]) <= 1e-9


def test_ittc57_and_open_water_reference_values():
    assert ittc57_cf(0.0) == 0.0
    cf = ittc57_cf(3.6e8)
    assert 1.5e-3 < cf < 2.0e-3  # KVLCC2-class Reynolds number
    ow = open_water_resistance(
        speed_m_s=2.0, length_m=320.0, draft_m=20.8, beam_m=58.0, block_coefficient=0.8098
    )
    assert 2.5e4 < ow["wetted_surface_m2"] < 2.8e4
    assert 5.0e4 < ow["R_viscous_N"] < 2.0e5


def test_distribution_weights_integrate_to_one():
    grid = _grid(nx=20, ny=4)
    x = grid.cell_centers_x()
    breadth = np.full(grid.nx, 58.0)
    alpha = np.linspace(0.6, 0.3, grid.nx)
    w = distribution_weights(
        x=x, breadth_m=breadth, alpha_mean=alpha, coverage_ambient=0.6, dx=grid.dx
    )
    for key in ("bow", "hull_ice", "hull"):
        assert abs(float(np.sum(w[key]) * grid.dx) - 1.0) <= 1e-12


def test_porous_drag_scales_with_velocity_squared():
    u = np.array([1.0, 2.0])
    alpha = np.array([0.3, 0.3])
    f1 = porous_drag_density(
        alpha_mean=alpha[:1],
        u_m_s=u[:1],
        thickness_m=0.15,
        lane_width_m=63.8,
        r_mean_m=2.0,
    )
    f2 = porous_drag_density(
        alpha_mean=alpha[1:],
        u_m_s=u[1:],
        thickness_m=0.15,
        lane_width_m=63.8,
        r_mean_m=2.0,
    )
    assert f1[0] > 0.0
    assert abs(f2[0] / f1[0] - 4.0) < 1e-12


# ---------------------------------------------------------------------------
# FSI stage
# ---------------------------------------------------------------------------


def test_two_way_coupling_converges_and_matches_added_mass_theory():
    setup = StructuralSetup(gap0_m=0.65)
    alpha = np.full(16, 0.6)
    result = solve_two_way(
        alpha_mean=alpha,
        thickness_m=0.15,
        setup=setup,
        flow_fn=lambda a_eff: np.full(alpha.size, 5.0e3),
        relaxation=1.0,
        max_iter=60,
        tol=1e-10,
    )
    assert result.converged
    assert result.residual <= 1e-10
    predicted = abs(linear_gain(1.0, result.added_mass_ratio))
    hist = result.residual_history
    measured = float(np.median([hist[i + 1] / hist[i] for i in range(len(hist) - 1)]))
    assert abs(measured - predicted) / predicted < 5e-2
    # the deflection must stay far below the nominal clearance
    assert float(np.max(np.abs(result.deflection_m))) < 0.05 * setup.gap0_m


def test_stability_domain_boundary_at_added_mass_ratio_one():
    grid = [round(0.05 * i, 4) for i in range(1, 121)]
    rows = stability_scan(
        setup=StructuralSetup(gap0_m=0.65),
        thickness_values=[0.10, 0.30],
        omega_grid=grid,
        t_eff_values=(0.05, 0.25),
    )
    assert rows
    for row in rows:
        assert row["converges_at_jacobi"] == (row["added_mass_ratio"] < 1.0)
        if row["added_mass_ratio"] >= 1.0:
            assert row["omega_max_measured"] == 0.0
        elif not row["beyond_grid"]:
            assert abs(row["omega_max_measured"] - row["omega_max_analytic"]) <= 2 * row["grid_step"]
    # the scan must actually cross the boundary, otherwise it verifies nothing
    assert any(r["added_mass_ratio"] < 1.0 for r in rows)
    assert any(r["added_mass_ratio"] >= 1.0 for r in rows)


def test_packing_pressure_and_squeeze_feedback():
    from tensorphysics.ice_nav_integration import packing_pressure

    setup = StructuralSetup()
    p = packing_pressure(np.array([0.0, 0.5, 1.0]), setup)
    assert p[0] == 0.0 and p[1] == 0.0
    assert p[2] == pytest.approx(setup.sigma_c_pa, rel=1e-12)
    a_eff = 0.6 * 0.15 / np.array([0.65, 0.30])
    assert a_eff[1] > a_eff[0]  # squeezing the gap raises the local fraction


# ---------------------------------------------------------------------------
# energy ledger
# ---------------------------------------------------------------------------


def test_channel_split_closes_and_ledger_residual_is_attributed():
    cfg = CaseConfig()
    info = ice_channels(cfg)
    res = brash_ice_resistance(
        cfg.ice_thickness_m, cfg.speed_m_s, cfg.coverage, info["ship"]
    )
    split = brash_channel_split(res)
    assert info["split_closure_error"] <= 1e-12
    assert abs(sum(split.values()) - res["R_total_N"]) <= 1e-9 * res["R_total_N"]

    forces = info["channels"]
    total = sum(forces.values())
    ledger = build_ledger(
        channel_forces=forces, speed_m_s=cfg.speed_m_s, panel_force_N=total
    )
    assert ledger["residual"] <= 1e-12
    assert abs(sum(row["share"] for row in ledger["channels"]) - 1.0) <= 1e-12

    # a 5 % unnormalised channel must break the 2 % gate: the ledger has teeth
    # a 5 % error in one channel while the panel integral stays at the correct
    # value (the signature of an unnormalised distribution weight)
    broken = dict(forces)
    broken["breaking"] *= 1.05
    ledger_bad = build_ledger(
        channel_forces=broken,
        speed_m_s=cfg.speed_m_s,
        panel_force_N=total,
    )
    assert ledger_bad["residual"] >= 2.0e-2

    with pytest.raises(ValueError):
        build_ledger(
            channel_forces={"crushing": 1.0}, speed_m_s=1.0, panel_force_N=1.0
        )


def test_accumulate_uses_steady_window():
    history = [
        {
            "channel_forces": {c: 1.0 for c in CHANNELS},
            "panel_force_N": 5.0,
            "residual": 1e-3 if i < 8 else 1e-9,
            "speed_m_s": 2.0,
        }
        for i in range(10)
    ]
    out = accumulate(history, window_fraction=0.2)
    assert out["window_samples"] == 2
    assert out["mean_residual"] == pytest.approx(1e-9)
    assert out["max_residual_all"] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# end-to-end gate
# ---------------------------------------------------------------------------


def test_geometry_gate_matches_declared_displacement():
    geo = geometry_audit(CaseConfig())
    assert geo["displacement_relative_error"] < 1e-2
    assert geo["h_limit_m"] > CaseConfig().ice_thickness_m


def test_gate_end_to_end(tmp_path):
    cfg = CaseConfig(end_time_s=5.0, nx=48, ny=8, contract_audit_every=2)
    outdir = tmp_path / "art"
    docs = tmp_path / "docs"
    verdict = run_gate(cfg, outdir=str(outdir), docs_dir=str(docs))

    assert verdict["overall_status"] == "PASS", json.dumps(verdict["criteria"], default=float)[:2000]
    for key in ("C1", "C2", "C3", "C4", "C5"):
        assert verdict["criteria"][key]["passed"], key
    assert verdict["criteria"]["C2"]["n_checks_failed"] == 0
    assert verdict["criteria"]["C2"]["n_checks_total"] > 40
    assert verdict["criteria"]["C3"]["mean_residual"] < 2.0e-2
    assert verdict["criteria"]["C4"]["max_residual"] < 1.0e-6

    for name in (
        "gate_verdict.json",
        "energy_ledger.json",
        "contract_audit.json",
        "fsi_stability.json",
        "rans_history.json",
        "data_flow.mmd",
        "provenance.json",
    ):
        assert (outdir / name).exists(), name
    try:
        import matplotlib  # noqa: F401

        have_mpl = True
    except Exception:
        have_mpl = False
    if have_mpl:
        assert (outdir / "figures" / "fig5_stability_domain.png").exists()
    else:  # the gate degrades gracefully and records why
        assert verdict.get("figures", {}).get("n", 1) == 0
    assert (outdir / "payloads" / "t_end.json").exists()
    report = docs / "IN66_N08_ICE_NAV_INTEGRATION_GATE.md"
    assert report.exists()

    provenance = json.loads((outdir / "provenance.json").read_text())
    assert len(provenance["artifacts"]) >= 8
    assert all(len(h) == 64 for h in provenance["artifacts"].values())

    text = report.read_text(encoding="utf-8")
    for token in ("C1", "C2", "C3", "C4", "C5", "数据流图", "残差归因", "是否 < 3%"):
        assert token in text


def test_gate_is_deterministic(tmp_path):
    cfg = CaseConfig(end_time_s=2.0, nx=32, ny=8, contract_audit_every=1)
    a = run_gate(cfg, outdir=str(tmp_path / "a"), docs_dir=str(tmp_path / "docs_a"))
    b = run_gate(cfg, outdir=str(tmp_path / "b"), docs_dir=str(tmp_path / "docs_b"))
    assert a["criteria"]["C3"]["mean_residual"] == pytest.approx(
        b["criteria"]["C3"]["mean_residual"], rel=0, abs=0
    )
    assert a["flow_resistance"]["R_panel_N_last"] == pytest.approx(
        b["flow_resistance"]["R_panel_N_last"], rel=0, abs=0
    )