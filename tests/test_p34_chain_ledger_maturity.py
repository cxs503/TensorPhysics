"""P34 chain-ledger maturity: three-stage serialisation conservation gates.

This test module validates the end-to-end chain:
  Stage 1 (LBM wave) → Stage 2 (6-DOF floating body) → Stage 3 (beam whipping)

with force-energy-time triple conservation at each hand-off face:
  A) LBM free-surface pressure → 6-DOF resultant force/moment
  B) 6-DOF displacement/velocity → LBM moving/immersed boundary
  C) Slamming pressure → beam modal generalised force (Duhamel)

Plus energy ledger (E), time synchronisation (T), and negative controls (N).

Mission: m7087 / P34-N08-P34-chain-ledger
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from tensorphysics.chain_ledger.readiness import CRITERIA, SCHEMA, GATE_ID, evaluate_chain
from tensorphysics.chain_ledger.runner import ChainConfig, run_chain_case


@pytest.fixture(scope="module")
def chain_detail(tmp_path_factory):
    """Run the full chain once for the module."""
    out = tmp_path_factory.mktemp("p34_chain")
    return run_chain_case(ChainConfig(), outdir=out)


# =====================================================================
# Top-level verdict
# =====================================================================
class TestChainVerdict:
    """The chain readiness verdict must be PASS and machine-readable."""

    def test_overall_pass(self, chain_detail):
        v = chain_detail["readiness"]
        assert v["overall_status"] == "PASS", f"blocking: {v['blocking_criteria']}"
        assert v["chain_ready"] is True
        assert v["schema"] == SCHEMA
        assert v["gate_id"] == GATE_ID

    def test_all_six_faces_pass(self, chain_detail):
        v = chain_detail["readiness"]
        for face in ("A", "B", "C", "E", "T", "N"):
            assert v["faces"][face]["verdict"] == "PASS", f"face {face} failed"
            assert v["faces"][face]["evidence_sha256"] is not None

    def test_no_missing_products(self, chain_detail):
        v = chain_detail["readiness"]
        assert v["missing_products"] == []
        assert v["blocking_criteria"] == []


# =====================================================================
# Face A: pressure → force (criterion a)
# =====================================================================
class TestFaceA:
    """∫p·n dA vs rigid-body resultant: rel err < 2%, order ≥ 1."""

    def test_force_relative_error_below_2pct(self, chain_detail):
        err = chain_detail["products"]["A"]["force_rel_err"]
        assert err < 0.02, f"Face A force error {err:.4e} >= 2%"

    def test_mesh_convergence_order_at_least_1(self, chain_detail):
        order = chain_detail["faces"]["A"]["observed_order"]
        assert order >= 1.0, f"Face A order {order:.3f} < 1"

    def test_monotone_convergence_under_refinement(self, chain_detail):
        rows = chain_detail["faces"]["A"]["rows"]
        errs = [r["hold_rel_err"] for r in rows]
        # monotonically decreasing under refinement
        assert errs == sorted(errs, reverse=True), "non-monotone convergence"

    def test_reference_self_consistency(self, chain_detail):
        ref = chain_detail["faces"]["A"]["reference_self_consistency"]
        assert ref < 0.02, f"reference self-consistency {ref:.4e} >= 2%"


# =====================================================================
# Face B: body motion → LBM boundary (mass/volume conservation)
# =====================================================================
class TestFaceB:
    """Volume/mass conservation and normal velocity consistency."""

    def test_closed_surface_zero_flux(self, chain_detail):
        flux = chain_detail["products"]["B"]["closed_flux_rel"]
        assert flux < 1e-8, f"Face B closed flux {flux:.4e} >= 1e-8"

    def test_volume_rate_identity(self, chain_detail):
        err = chain_detail["products"]["B"]["volume_rel_err"]
        assert err < 0.02, f"Face B volume rate error {err:.4e} >= 2%"

    def test_normal_velocity_consistency(self, chain_detail):
        err = chain_detail["products"]["B"]["normal_velocity_rel_err"]
        assert err < 0.02, f"Face B normal velocity error {err:.4e} >= 2%"

    def test_power_reciprocity(self, chain_detail):
        err = chain_detail["products"]["B"]["power_rel_err"]
        assert err < 0.02, f"Face B power reciprocity error {err:.4e} >= 2%"

    def test_negative_control_breaks_flux(self, chain_detail):
        face_b = chain_detail["faces"]["B"]
        assert face_b["closed_flux_negative_control_rel"] > 1e-3


# =====================================================================
# Face C: slamming → modal force (criterion b)
# =====================================================================
class TestFaceC:
    """Modal generalised force vs full-field virtual work < 3%."""

    def test_modal_force_relative_error_below_3pct(self, chain_detail):
        err = chain_detail["products"]["C"]["modal_force_rel_err"]
        assert err < 0.03, f"Face C modal force error {err:.4e} >= 3%"

    def test_duhamel_vs_rk4_below_3pct(self, chain_detail):
        err = chain_detail["products"]["C"]["duhamel_rel_err"]
        assert err < 0.03, f"Face C Duhamel error {err:.4e} >= 3%"

    def test_peak_whipping_stress_physically_plausible(self, chain_detail):
        face_c = chain_detail["faces"]["C"]
        peak = face_c["peak_whipping_stress_MPa"]
        assert 0.05 < peak < 250.0, f"peak stress {peak} MPa unphysical"

    def test_modal_truncation_audit(self, chain_detail):
        face_c = chain_detail["faces"]["C"]
        trunc = face_c["peak_stress_by_mode_count_Pa"]
        rel = abs(trunc["8"] - trunc["6"]) / trunc["8"]
        assert rel < 0.15, f"modal truncation {rel:.3f} >= 15%"


# =====================================================================
# Energy ledger (criterion c)
# =====================================================================
class TestEnergyLedger:
    """W_in - (dKE + dPE + W_diss) residual / W_in < 3%."""

    def test_wave_energy_ledger_below_3pct(self, chain_detail):
        err = chain_detail["products"]["E"]["wave_ledger_rel_err"]
        assert err < 0.03, f"wave energy ledger error {err:.4e} >= 3%"

    def test_body_energy_ledger_below_3pct(self, chain_detail):
        err = chain_detail["products"]["E"]["body_ledger_rel_err"]
        assert err < 0.03, f"body energy ledger error {err:.4e} >= 3%"

    def test_wave_input_work_positive(self, chain_detail):
        w_in = chain_detail["faces"]["E"]["wave"]["W_in_J"]
        assert w_in > 0.0, "wave input work must be positive"

    def test_marched_body_matches_steady_state(self, chain_detail):
        amp_diff = chain_detail["faces"]["E"]["marched"]["amplitude_rel_diff"]
        assert amp_diff < 0.02, f"marched amplitude diff {amp_diff:.4e} >= 2%"


# =====================================================================
# Time synchronisation (criterion d)
# =====================================================================
class TestTimeLedger:
    """Zero-offset residual < 3%; Δt/2 offset detected (>10%)."""

    def test_zero_offset_residual_below_3pct(self, chain_detail):
        err = chain_detail["products"]["T"]["zero_offset_rel_err"]
        assert err < 0.03, f"zero-offset error {err:.4e} >= 3%"

    def test_half_step_offset_detected(self, chain_detail):
        residual = chain_detail["products"]["T"]["offset_residual"]
        assert residual > 0.10, f"offset residual {residual:.4e} <= 10%"

    def test_zero_order_force_error_above_10pct(self, chain_detail):
        err = chain_detail["products"]["T"]["zero_order_force_err"]
        assert err > 0.10, f"zero-order error {err:.4e} <= 10%"

    def test_detectors_fire(self, chain_detail):
        assert chain_detail["products"]["T"]["offset_detected"] == 1.0
        assert chain_detail["products"]["T"]["zero_order_detected"] == 1.0


# =====================================================================
# Negative controls (criterion e)
# =====================================================================
class TestNegativeControls:
    """No stabilisation → divergence 100%; 0th order → >10% force error."""

    def test_divergence_detection_rate_100pct(self, chain_detail):
        rate = chain_detail["products"]["N"]["divergence_detection_rate"]
        assert rate == 1.0, f"divergence detection rate {rate} != 100%"

    def test_stabilised_convergence_rate_100pct(self, chain_detail):
        rate = chain_detail["products"]["N"]["stabilised_converged_rate"]
        assert rate == 1.0, f"stabilised convergence rate {rate} != 100%"

    def test_spectral_radius_indicates_instability(self, chain_detail):
        rho = chain_detail["faces"]["N"]["spectral_radius"]
        assert rho > 1.0, f"spectral radius {rho:.4f} <= 1 (should be unstable)"

    def test_mass_ratio_in_light_body_regime(self, chain_detail):
        mr = chain_detail["faces"]["N"]["mass_ratio"]
        assert mr > 0.5, f"mass ratio {mr:.4f} <= 0.5"

    def test_added_mass_dominates_structural(self, chain_detail):
        ar = chain_detail["faces"]["N"]["added_mass_ratio"]
        assert ar < 1.0, f"added mass ratio {ar:.4f} >= 1.0"


# =====================================================================
# Falsifiability (criterion f)
# =====================================================================
class TestFalsifiability:
    """Dropping a face → WITHHELD; breaking a threshold → FAIL."""

    def test_missing_face_withholds(self, chain_detail):
        products = {k: dict(v) for k, v in chain_detail["products"].items()}
        for face in ("A", "B", "C"):
            products[face] = None
            verdict = evaluate_chain(products)
            assert verdict["overall_status"] == "WITHHELD"
            assert face in verdict["missing_products"]

    def test_broken_threshold_fails(self, chain_detail):
        products = {k: dict(v) for k, v in chain_detail["products"].items()}
        products["E"]["wave_ledger_rel_err"] = 0.5
        verdict = evaluate_chain(products)
        assert verdict["overall_status"] == "FAIL"
        assert "E:wave_ledger_rel_err" in verdict["blocking_criteria"]


# =====================================================================
# Geometry & physics sanity
# =====================================================================
class TestGeometrySanity:
    """Cb, L, B consistency; dispersion relation; wave energy flux."""

    def test_dispersion_relation_satisfied(self, chain_detail):
        assert chain_detail["wave"]["dispersion_residual"] < 1e-12

    def test_numerical_energy_flux_matches_analytic(self, chain_detail):
        w = chain_detail["wave"]
        assert w["numerical_energy_flux_W_m"] == pytest.approx(
            w["energy_flux_W_m"], rel=1e-2
        )

    def test_geometry_parameters_consistent(self, chain_detail):
        g = chain_detail["geometry"]
        # displacement = rho * L * B * T (box body)
        expected_disp = 1025.0 * g["length_m"] * g["breadth_m"] * g["draught_m"]
        assert g["displacement_kg"] == pytest.approx(expected_disp, rel=1e-6)

    def test_effective_mass_matrix_symmetric(self, chain_detail):
        m = np.asarray(chain_detail["hydrodynamics"]["effective_mass_matrix"])
        np.testing.assert_allclose(m, m.T, rtol=0, atol=0)
