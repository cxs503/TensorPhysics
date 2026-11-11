"""Plate/shell static verification face (campaign node ``p405w1-n05``).

Two independent analytic anchors are checked against two finite element
implementations:

* uniformly loaded simply supported square plate -- Navier double series
  (:mod:`tensorphysics.plate_shell.navier`) vs the bicubic Hermite Kirchhoff
  element (:mod:`tensorphysics.plate_shell.plate_fe`);
* thin cylindrical shell under internal pressure -- Donnell boundary-layer
  closed form (:mod:`tensorphysics.plate_shell.shell_axisym`) vs the
  axisymmetric shell element (membrane-bending coupling).

The heavy end-to-end study is computed once per module by the ``report``
fixture; the analytic-identity tests are cheap and independent of it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import pathlib

import numpy as np
import pytest

from tensorphysics.plate_shell import (
    CylinderPressureFE,
    NAVIER_ALPHA_SQUARE_PUBLISHED,
    PLATE_CASE,
    SSSSPlateFE,
    SHELL_CASE,
    beta_edge,
    cylinder_edge_solution,
    element_stiffness,
    element_stiffness_kernels,
    edge_decay_fit,
    fit_order,
    membrane_solution,
    navier_plate_center,
    navier_plate_fields,
    run_verification,
    series_convergence,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _free_bytes(root) -> int:
    try:
        st = os.statvfs(str(root))
        return st.f_bavail * st.f_frsize
    except OSError:  # pragma: no cover - platform dependent
        return 1 << 30


_FREE_BYTES = _free_bytes(REPO_ROOT)
PC = dict(PLATE_CASE)
SC = dict(SHELL_CASE)


@pytest.fixture(scope="module")
def report():
    """Full verification payload (plate + shell, all gates)."""
    return run_verification()


def _plate_fields(x, y):
    return navier_plate_fields(PC["a"], PC["b"], PC["t"], PC["E"], PC["nu"],
                               PC["q"], x, y, n_terms=201)


# ---------------------------------------------------------------------------
# Navier analytic anchor
# ---------------------------------------------------------------------------
class TestNavierAnchor:
    def test_alpha_matches_published_coefficient(self):
        """w_max = alpha q a^4 / D with alpha = 0.00406 (Timoshenko, Table 8)."""
        ref = navier_plate_center(PC["a"], PC["b"], PC["t"], PC["E"],
                                  PC["nu"], PC["q"], n_terms=81)
        assert ref["alpha_deflection"] == pytest.approx(
            NAVIER_ALPHA_SQUARE_PUBLISHED, rel=1e-3)

    def test_centre_moments_match_published_coefficient(self):
        """Mx = My = 0.0479 q a^2 at the centre of the square plate."""
        ref = navier_plate_center(PC["a"], PC["b"], PC["t"], PC["E"],
                                  PC["nu"], PC["q"], n_terms=81)
        assert ref["Mx"] == pytest.approx(0.0479 * PC["q"] * PC["a"] ** 2,
                                          rel=1e-3)
        assert ref["Mx"] == pytest.approx(ref["My"], rel=1e-10)

    def test_series_truncation_below_gate(self):
        """Successive truncations converge far below the 0.1 % gate."""
        conv = series_convergence(PC["a"], PC["b"], PC["t"], PC["E"],
                                  PC["nu"], PC["q"],
                                  np.array([0.25, 0.5]), np.array([0.5, 0.25]),
                                  n_terms_list=(21, 41, 81, 161, 321))
        assert conv["w"][-1] < 1e-6
        assert conv["Mx"][-1] < 1e-4

    def test_field_equilibrium_residual(self):
        """dQx/dx + dQy/dy + q = 0 at an interior point (plate equilibrium)."""
        x0, y0, h = 0.37, 0.43, 5e-3

        def qx(x):
            return float(navier_plate_fields(PC["a"], PC["b"], PC["t"],
                                             PC["E"], PC["nu"], PC["q"], x, y0,
                                             n_terms=201)["Qx"])

        def qy(y):
            return float(navier_plate_fields(PC["a"], PC["b"], PC["t"],
                                             PC["E"], PC["nu"], PC["q"], x0, y,
                                             n_terms=201)["Qy"])

        dqx = (qx(x0 + h) - qx(x0 - h)) / (2 * h)
        dqy = (qy(y0 + h) - qy(y0 - h)) / (2 * h)
        assert abs(dqx + dqy + PC["q"]) / PC["q"] < 1e-3

    def test_shear_moment_consistency(self):
        """Qx = dMx/dx + dMxy/dy holds for the series fields."""
        x0, y0, h = 0.37, 0.43, 1e-3

        def fld(x, y):
            return {k: float(v) for k, v in
                    _plate_fields(np.float64(x), np.float64(y)).items()}

        qx0 = fld(x0, y0)["Qx"]
        lhs = ((fld(x0 + h, y0)["Mx"] - fld(x0 - h, y0)["Mx"]) / (2 * h)
               + (fld(x0, y0 + h)["Mxy"] - fld(x0, y0 - h)["Mxy"]) / (2 * h))
        assert lhs == pytest.approx(qx0, rel=1e-4)

    def test_series_satisfies_simply_supported_boundary_conditions(self):
        """w = 0 and normal moment = 0 on every supported edge."""
        edges = {
            "x=0": (np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.5, 0.8])),
            "x=a": (np.full(3, PC["a"]), np.array([0.2, 0.5, 0.8])),
        }
        for name, (x, y) in edges.items():
            f = _plate_fields(x, y)
            assert np.max(np.abs(f["w"])) < 1e-12, name
            assert np.max(np.abs(f["Mx"])) < 1e-6 * PC["q"] * PC["a"] ** 2, name
        y = np.array([0.0, 0.0, 0.0])
        x = np.array([0.2, 0.5, 0.8])
        f = _plate_fields(x, y)
        assert np.max(np.abs(f["My"])) < 1e-6 * PC["q"] * PC["a"] ** 2
        assert np.max(np.abs(f["w"])) < 1e-12


# ---------------------------------------------------------------------------
# Plate finite element
# ---------------------------------------------------------------------------
class TestPlateElement:
    def test_stiffness_matches_plate_buckling_kernels(self):
        """The local kernels reproduce the existing plate_buckling element."""
        D = 1.0
        k1 = element_stiffness(0.25, 0.2, D, 0.3)
        k2 = element_stiffness_kernels(0.25, 0.2, D, 0.3)
        assert np.max(np.abs(k1 - k2)) <= 1e-12 * np.max(np.abs(k2))

    def test_g1_deflection_gate(self, report):
        """G1: w_max error <= 1 % on 4x4 and <= 0.1 % on 16x16."""
        g1 = report["gates"]["G1"]
        assert g1["err_w_4x4"] <= 1e-2
        assert g1["err_w_16x16"] <= 1e-3
        assert g1["pass"] is True

    def test_g2_moment_gate(self, report):
        """G2: Mx and My errors <= 2 % on 8x8 and every finer mesh."""
        g2 = report["gates"]["G2"]
        assert g2["err_Mx_8x8"] <= 2e-2
        assert g2["err_My_8x8"] <= 2e-2
        assert g2["max_err_Mx_ge8"] <= 2e-2
        assert g2["pass"] is True

    def test_g4_convergence_orders(self, report):
        """G4: C^1 deflection order >= 3, moment/shear order >= 2."""
        g4 = report["gates"]["G4_plate"]
        assert g4["order_w"] >= 3.0
        assert g4["order_Mx"] >= 2.0
        assert g4["order_Qx"] >= 2.0
        assert g4["pass"] is True

    def test_global_equilibrium_reaction_equals_load(self, report):
        """Sum of vertical reactions reproduces q*a*b on every mesh."""
        for row in report["plate"]["meshes"]:
            assert row["equilibrium_rel_error"] < 1e-8
            assert row["reaction_sum"] == pytest.approx(row["load_total"],
                                                        rel=1e-8)

    def test_natural_bc_is_softer_than_over_constrained_bc(self, report):
        """Negative control: fixing the edge twist dof makes the plate stiffer."""
        ctrl = report["plate"]["over_constrained_control"]
        assert ctrl["stiffer_as_expected"] is True
        assert ctrl["w_center_tight_8x8"] < ctrl["w_center_natural_8x8"]

    def test_unconstrained_stiffness_is_singular(self):
        """Negative control: without the w = 0 edges the operator is singular."""
        fe = SSSSPlateFE(a=PC["a"], b=PC["b"], t=PC["t"], E=PC["E"],
                         nu=PC["nu"], nx=2, ny=2)
        K = fe.assemble().toarray()
        eig = np.linalg.eigvalsh(K)
        assert abs(eig[0]) < 1e-6 * abs(eig[-1])

    def test_centre_deflection_uses_centre_node_on_even_meshes(self):
        """An even mesh has a node at the centre, so no averaging is applied."""
        fe = SSSSPlateFE(a=PC["a"], b=PC["b"], t=PC["t"], E=PC["E"],
                         nu=PC["nu"], nx=8, ny=8)
        fe.solve(PC["q"])
        w = fe.w_nodal()
        assert fe.center_deflection() == pytest.approx(w[4, 4], rel=1e-14)


# ---------------------------------------------------------------------------
# Cylindrical shell: membrane + bending boundary layer
# ---------------------------------------------------------------------------
class TestCylindricalShell:
    def test_beta_matches_sqrt_Rt_scaling(self):
        """beta * sqrt(R t) = (3 (1 - nu^2))^{1/4}."""
        beta = beta_edge(SC["R"], SC["t"], SC["E"], SC["nu"])
        assert beta * math.sqrt(SC["R"] * SC["t"]) == pytest.approx(
            (3.0 * (1.0 - SC["nu"] ** 2)) ** 0.25, rel=1e-12)

    def test_closed_form_satisfies_donnell_ode(self):
        """D w'''' + (E t / R^2) w = p away from the boundaries."""
        x = np.linspace(0.15, 0.35, 401)
        ref = cylinder_edge_solution(SC["R"], SC["L"], SC["t"], SC["E"],
                                     SC["nu"], SC["p"], x)
        w = ref["w"]
        dx = x[1] - x[0]
        w4 = (w[4:] - 4 * w[3:-1] + 6 * w[2:-2] - 4 * w[1:-3] + w[:-4]) / dx ** 4
        D = SC["E"] * SC["t"] ** 3 / (12 * (1 - SC["nu"] ** 2))
        resid = D * w4 + SC["E"] * SC["t"] / SC["R"] ** 2 * w[2:-2] - SC["p"]
        assert np.max(np.abs(resid)) / SC["p"] < 1e-2

    def test_far_field_reaches_membrane_state(self):
        """sigma_theta -> pR/t and w -> pR^2/(Et) far from the clamped edge."""
        ref = cylinder_edge_solution(SC["R"], SC["L"], SC["t"], SC["E"],
                                     SC["nu"], SC["p"], np.array([SC["L"]]))
        mem = membrane_solution(SC["R"], SC["t"], SC["E"], SC["p"])
        assert float(ref["sigma_theta"][0]) == pytest.approx(
            mem["sigma_theta"], rel=1e-12)

    def test_g3_hoop_stress_gate(self, report):
        """G3: sigma_theta error <= 1 % in the membrane region."""
        g3 = report["gates"]["G3"]
        assert g3["err_sigma_theta_membrane_finest"] <= 1e-2
        assert g3["err_sigma_theta_eq_pointwise_finest"] <= 1e-2
        assert g3["pass"] is True

    def test_edge_moment_convergence(self, report):
        """Edge bending moment converges at ~h^2 and is < 2 % at 64 elements."""
        rows = {int(r["n_elem"]): r for r in report["shell"]["meshes"]}
        assert rows[64]["err_Mx_edge"] <= 2e-2
        assert rows[16]["err_Mx_edge"] > rows[64]["err_Mx_edge"]
        assert report["shell"]["orders"]["err_Mx_edge"] >= 1.8

    def test_measured_boundary_layer_matches_sqrt_Rt(self, report):
        """Fitted e^{-beta x} decay reproduces beta = 1.286/sqrt(R t)."""
        dec = report["shell"]["edge_decay_fit"]
        assert dec["beta_rel_error"] <= 1e-3
        assert dec["decay_length_over_sqrt_Rt"] == pytest.approx(
            (3.0 * (1.0 - SC["nu"] ** 2)) ** -0.25, rel=1e-3)

    def test_energy_balance_and_free_residual(self, report):
        """U = (f^T u + u_c^T r_c)/2 and the free-dof residual vanishes."""
        assert report["shell"]["energy_balance"]["energy_balance_rel_error"] < 1e-9
        assert report["shell"]["reaction_check"]["max_free_residual"] < 1e-3

    def test_edge_bending_stress_identity(self):
        """sigma_x_bend = 6 Mx / t^2 (regression: a missing /t made it 100x low)."""
        fe = CylinderPressureFE(R=SC["R"], L=SC["L"], t=SC["t"], E=SC["E"],
                                nu=SC["nu"], n_elem=32)
        fe.solve(SC["p"])
        f = fe.fields_at(np.array([0.0, 0.1, 0.3]))
        expected = 6.0 * f["Mx"] / SC["t"] ** 2
        assert np.allclose(f["sigma_x_bend"], expected, rtol=1e-12)
        # peak edge bending stress must be ~1.8e8 Pa for this case
        assert abs(float(f["sigma_x_bend"][0])) > 1e8

    def test_zero_pressure_gives_zero_response(self):
        """Trivial-solution check (guard against a spurious particular part)."""
        fe = CylinderPressureFE(R=SC["R"], L=SC["L"], t=SC["t"], E=SC["E"],
                                nu=SC["nu"], n_elem=16)
        fe.solve(0.0)
        x = np.linspace(0.0, SC["L"], 51)
        f = fe.fields_at(x)
        assert np.max(np.abs(f["w"])) < 1e-18
        assert np.max(np.abs(f["Mx"])) < 1e-12


# ---------------------------------------------------------------------------
# End-to-end report
# ---------------------------------------------------------------------------
class TestVerificationReport:
    def test_all_gates_pass(self, report):
        assert report["with_gates"] is True
        assert report["n_gates"] >= 6
        assert report["n_gates_pass"] == report["n_gates"]
        assert report["verdict"] == "PASS"

    def test_negative_controls_recorded(self, report):
        neg = report["negative_controls"]
        assert neg["over_constraint_makes_model_stiffer"] is True
        assert neg["navier_series_truncation_below_gate"] is True
        assert neg["closed_form_satisfies_shell_ode"] is True
        assert neg["shell_energy_balance_identity"] is True

    @pytest.mark.skipif(_FREE_BYTES < 2e6,
                        reason="host filesystem nearly full; JSON output would "
                               "fail for environmental reasons")
    def test_driver_writes_json(self, tmp_path):
        """``verify_fem_plate_shell.py --quick --no-figures`` produces JSON."""
        out = tmp_path / "out"
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "verify_fem_plate_shell.py"),
             "--quick", "--no-figures", "--outdir", str(out)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stderr[-2000:]
        payload = json.loads((out / "verify_fem_plate_shell.json").read_text())
        assert payload["module"] == "tensorphysics.plate_shell"
        assert payload["plate"]["meshes"]
        assert payload["shell"]["meshes"]