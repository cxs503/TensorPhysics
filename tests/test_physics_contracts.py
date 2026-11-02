"""Physics contract registry tests (R1, 2026-11-02 task book stage 2).

Anchors come from audit m2086 crosscheck artifacts (11 audited kernels), NOT
re-derived here: SUBOFF-01 input {L=4.356, B=0.508, T=0.508, Cb=0.542,
V=3.0, rho=998.2, nu=1.004e-6} with empirical wetted surface; the shared
composition reproduced ct = 0.003461889103158906 bitwise in 6 of 11 kernels.
"""
import math

import pytest

from tensorphysics.physics_contracts import (
    PhysicsContractError,
    PhysicsModelContract,
    ReferenceBand,
    get_contract,
    all_contracts,
    SUBOFF_SHARED_CORE_LABEL,
)
from tensorphysics.physics_kernels_resistance import (
    axisymmetric_wetted_surface,
    ittc_1957_cf,
    schoenherr_1932_cf,
    shared_core_resistance,
    suboff_residual_coefficient,
)

SUBOFF = dict(length_m=4.356, beam_m=0.508, cb=0.542, speed_ms=3.0,
              rho_kg_m3=998.2, nu_m2_s=1.004e-06,
              wetted_surface_m2=4.417911491407555)  # = axisymmetric fit, declared


class TestKernels:
    def test_suboff01_bitwise_anchor(self):
        """Shared core must reproduce the 6-kernel bitwise agreement value."""
        r = shared_core_resistance(**SUBOFF)
        assert r.ct == pytest.approx(0.003461889103158906, rel=1e-9)
        assert r.rt_n == pytest.approx(68.7005544395378, rel=1e-9)
        assert r.cf == pytest.approx(0.0028672071638755144, rel=1e-9)
        assert r.cr == pytest.approx(0.0005946819392833916, rel=1e-9)
        assert r.reynolds == pytest.approx(13015936.25, rel=1e-6)

    def test_wetted_surface_is_required_no_silent_fallback(self):
        """The audited +50.6% KVLCC2 failure mode must be impossible here."""
        bad = dict(SUBOFF)
        bad["wetted_surface_m2"] = None
        with pytest.raises(ValueError, match="wetted_surface_m2"):
            shared_core_resistance(**bad)

    def test_kvlcc2_empirical_bias_documented(self):
        """Empirical S on KVLCC2 must land in the audited +50.6% bias band."""
        s = axisymmetric_wetted_surface(320.0, 58.0, 0.8098)
        assert s == pytest.approx(40958.0, rel=5e-3)
        assert 1.49 <= s / 27194.0 <= 1.52  # +50.6% vs published

    def test_ittc_vs_schoenherr_independent_crosscheck(self):
        """Two distinct correlations: gap shrinks with Re, crossing ~6e8.

        Measured 2026-11-02 on this module (v1 band assertion 2-6% was wrong
        at both ends -- unmeasured assertion removed as anti-pattern):
        rel(ittc-schoenherr)/ittc = +5.93e-2, +2.19e-2, +5.43e-3, -2.1e-4
        at Re = 1e6, 1e7, 1e8, 1e9.
        """
        measured = {1e6: 5.93e-2, 1e7: 2.19e-2, 1e8: 5.43e-3, 1e9: -2.1e-4}
        prev = None
        for re_ in sorted(measured):
            a, b = ittc_1957_cf(re_), schoenherr_1932_cf(re_)
            rel = (a - b) / a
            assert rel == pytest.approx(measured[re_], abs=2e-3)
            if prev is not None:
                assert rel < prev  # monotonically converging with Re
            prev = rel

    def test_residual_fit_fn_floor(self):
        assert suboff_residual_coefficient(0.04, 0.542) == 0.0
        assert suboff_residual_coefficient(0.5, 0.542) > 0.0

    def test_input_validation(self):
        with pytest.raises(ValueError):
            ittc_1957_cf(-1.0)
        with pytest.raises(ValueError):
            shared_core_resistance(**{**SUBOFF, "form_factor": 0.9})


class TestNamingGate:
    def test_gate_rejects_holtrop_mennen_label_on_shared_core(self):
        """The audited mislabel must fail at construction, not at review."""
        with pytest.raises(PhysicsContractError, match="naming violation"):
            PhysicsModelContract(
                key="resistance.mislabeled",
                declared_method="simplified Holtrop-Mennen formulation for SUBOFF",
                inputs=(get_contract("resistance.suboff_shared_core").inputs),
                components=get_contract("resistance.suboff_shared_core").components,
                reference_kernel="tensorphysics.physics_kernels_resistance."
                                 "shared_core_resistance",
            )

    def test_gate_accepts_full_holtrop1984(self):
        c = get_contract("resistance.holtrop1984_full")
        assert "Holtrop" in c.declared_method and "full" in c.declared_method

    def test_registered_label_is_truthful(self):
        assert "holtrop" not in SUBOFF_SHARED_CORE_LABEL.lower()

    def test_all_registered_contracts_valid(self):
        assert len(all_contracts()) == 5

    def test_alias_resolution_covers_audited_kernels(self):
        for a in ("A_endpoint_inline", "B_ship_resistance", "F_resistance_holtrop",
                  "G_holtrop1984", "D2_domain_same_wetted_surface",
                  "J_endpoint_k115"):
            assert get_contract(a) is not None

    def test_unknown_contract_raises(self):
        with pytest.raises(PhysicsContractError, match="unknown physics contract"):
            get_contract("resistance.no_such_thing")


class TestReferenceBands:
    def test_withheld_cannot_be_judgment_anchor(self):
        with pytest.raises(PhysicsContractError, match="WITHHELD"):
            ReferenceBand("X", "ct", 0.4, "-", provenance="WITHHELD",
                          tolerance_rel=0.15, admissible_for_judgment=True)

    def test_reference_provenance_recorded(self):
        c = get_contract("resistance.suboff_shared_core")
        deriv = [r for r in c.references if r.provenance == "DERIVED"]
        exp = [r for r in c.references if r.provenance == "EXPERIMENTAL"]
        assert deriv and exp
        # the 15% gate hangs on the EXPERIMENTAL anchor, not the self-agreement
        assert any(r.tolerance_rel == 0.15 and r.admissible_for_judgment
                   for r in exp)
