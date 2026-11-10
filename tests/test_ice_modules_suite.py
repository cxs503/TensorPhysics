"""Test suite for ice_modules: 8 categories × 5 conditions = 40 verification points.

Covers:
- C1: level_ice 5 conditions error < 15%
- C2: local_pressure 5 conditions error < 15%
- C3: ice_plate 5 conditions error < 5%
- C4: broken_ice 5 conditions error < 15%
- C5: ice_ridge 5 conditions error < 15%
- C6: ice_abrasion/ice_adhesion/ice_accumulation 5 conditions each error < 15%
- C7: 40 verification points all green
"""

from __future__ import annotations
import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.ice_modules import run_all_suites, ALL_MODULES
from tensorphysics.ice_modules.level_ice import LevelIceModel, run_level_ice_suite
from tensorphysics.ice_modules.local_pressure import LocalPressureModel, run_local_pressure_suite
from tensorphysics.ice_modules.ice_plate import IcePlateModel, run_ice_plate_suite
from tensorphysics.ice_modules.broken_ice import BrokenIceModel, run_broken_ice_suite
from tensorphysics.ice_modules.ice_ridge import IceRidgeModel, run_ice_ridge_suite
from tensorphysics.ice_modules.ice_abrasion import IceAbrasionModel, run_ice_abrasion_suite
from tensorphysics.ice_modules.ice_adhesion import IceAdhesionModel, run_ice_adhesion_suite
from tensorphysics.ice_modules.ice_accumulation import IceAccumulationModel, run_ice_accumulation_suite


# ── C1: Level ice resistance ──────────────────────────────────────────
class TestLevelIce:
    """C1: Lindqvist/Keinonen/Riska level ice resistance, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_level_ice_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_level_ice_suite()
        for r in results:
            assert r["error_pct"] < 15.0, f"{r['condition']}: {r['error_pct']:.1f}% >= 15%"

    def test_physics_positive_resistance(self):
        model = LevelIceModel()
        for h in [0.5, 0.7, 0.9, 1.1, 1.5]:
            r = model.resistance(h, 3.0)
            assert r["R_total"] > 0
            assert r["R_crush"] > 0
            assert r["R_break"] > 0
            assert r["R_submersion"] > 0

    def test_resistance_increases_with_thickness(self):
        model = LevelIceModel()
        prev = 0
        for h in [0.5, 0.7, 0.9, 1.1, 1.5]:
            r = model.resistance(h, 3.0)
            assert r["R_total"] > prev
            prev = r["R_total"]


# ── C2: Local pressure ────────────────────────────────────────────────
class TestLocalPressure:
    """C2: IACS UR I pressure-area law, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_local_pressure_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_local_pressure_suite()
        for r in results:
            assert r["error_pct"] < 15.0

    def test_pressure_decreases_with_area(self):
        model = LocalPressureModel()
        prev_p = float('inf')
        for a in [1, 5, 10, 25, 50]:
            r = model.pressure(a, 1.0)
            assert r["P_MPa"] < prev_p
            prev_p = r["P_MPa"]


# ── C3: Ice plate (Hetenyi) ──────────────────────────────────────────
class TestIcePlate:
    """C3: Hetenyi elastic plate, 5 conditions, error < 5%."""

    def test_suite_all_pass(self):
        results = run_ice_plate_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds_strict(self):
        results = run_ice_plate_suite()
        for r in results:
            assert r["error_pct"] < 5.0, f"{r['condition']}: {r['error_pct']:.4f}% >= 5%"

    def test_deflection_positive(self):
        model = IcePlateModel()
        for case in [{"P": 10000, "h": 0.5}, {"P": 50000, "h": 1.0}, {"P": 200000, "h": 1.5}]:
            r = model.deflection(case["P"], case["h"])
            assert r["w_m"] > 0
            assert r["sigma_Pa"] > 0


# ── C4: Broken ice ────────────────────────────────────────────────────
class TestBrokenIce:
    """C4: Kotovirta broken ice field, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_broken_ice_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_broken_ice_suite()
        for r in results:
            assert r["error_pct"] < 15.0

    def test_resistance_increases_with_ridge_density(self):
        model = BrokenIceModel()
        prev = 0
        for rpk in [2, 4, 6, 8, 10]:
            r = model.resistance(rpk, 3.0)
            assert r["R_total"] > prev
            prev = r["R_total"]


# ── C5: Ice ridge ─────────────────────────────────────────────────────
class TestIceRidge:
    """C5: Riska ice ridge loads, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_ice_ridge_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_ice_ridge_suite()
        for r in results:
            assert r["error_pct"] < 15.0

    def test_load_increases_with_keel_depth(self):
        model = IceRidgeModel()
        prev = 0
        for hk in [2, 4, 6, 8, 10]:
            r = model.ridge_load(hk)
            assert r["F_total"] > prev
            prev = r["F_total"]


# ── C6a: Ice abrasion ─────────────────────────────────────────────────
class TestIceAbrasion:
    """C6: Su ice abrasion, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_ice_abrasion_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_ice_abrasion_suite()
        for r in results:
            assert r["error_pct"] < 15.0

    def test_wear_increases_with_speed(self):
        model = IceAbrasionModel()
        prev = 0
        for v in [1, 2, 3, 4, 5]:
            r = model.wear_rate(v, 100.0)
            assert r["wear_depth_mm"] > prev
            prev = r["wear_depth_mm"]


# ── C6b: Ice adhesion ─────────────────────────────────────────────────
class TestIceAdhesion:
    """C6: Mason ice adhesion, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_ice_adhesion_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_ice_adhesion_suite()
        for r in results:
            assert r["error_pct"] < 15.0

    def test_adhesion_peaks_near_optimal_temperature(self):
        model = IceAdhesionModel()
        # Adhesion should peak near T_opt = -10°C
        tau_minus10 = model.adhesion_strength(-10.0)["tau_kPa"]
        tau_minus5 = model.adhesion_strength(-5.0)["tau_kPa"]
        tau_minus25 = model.adhesion_strength(-25.0)["tau_kPa"]
        assert tau_minus10 >= tau_minus5
        assert tau_minus10 >= tau_minus25


# ── C6c: Ice accumulation ─────────────────────────────────────────────
class TestIceAccumulation:
    """C6: Bredmoes ice accumulation, 5 conditions, error < 15%."""

    def test_suite_all_pass(self):
        results = run_ice_accumulation_suite()
        assert len(results) == 5
        for r in results:
            assert r["pass"], f"FAIL: {r['condition']} error={r['error_pct']:.1f}%"

    def test_error_bounds(self):
        results = run_ice_accumulation_suite()
        for r in results:
            assert r["error_pct"] < 15.0

    def test_accumulation_increases_with_colder_temp(self):
        model = IceAccumulationModel()
        prev = 0
        for T in [-10, -15, -20, -25, -30]:
            r = model.accumulation_rate(T, 15.0)
            assert r["rate_mm_per_hour"] > prev
            prev = r["rate_mm_per_hour"]


# ── C7: All 40 verification points ────────────────────────────────────
class TestAllModules:
    """C7: 8 categories × 5 conditions = 40 verification points all green."""

    def test_total_40_points(self):
        results = run_all_suites()
        total = sum(len(v) for v in results.values())
        assert total == 40, f"Expected 40 verification points, got {total}"

    def test_all_40_pass(self):
        results = run_all_suites()
        for mod, cases in results.items():
            for c in cases:
                assert c["pass"], f"FAIL: {mod}/{c['condition']} error={c['error_pct']:.1f}%"

    def test_8_modules_present(self):
        results = run_all_suites()
        expected = {"level_ice", "local_pressure", "ice_plate", "broken_ice",
                    "ice_ridge", "ice_abrasion", "ice_adhesion", "ice_accumulation"}
        assert set(results.keys()) == expected

    def test_each_module_has_5_conditions(self):
        results = run_all_suites()
        for mod, cases in results.items():
            assert len(cases) == 5, f"{mod} has {len(cases)} conditions, expected 5"

    def test_max_error_under_15_percent(self):
        results = run_all_suites()
        max_err = 0
        for mod, cases in results.items():
            for c in cases:
                if c["error_pct"] > max_err:
                    max_err = c["error_pct"]
        assert max_err < 15.0, f"Max error {max_err:.1f}% >= 15%"

    def test_ice_plate_under_5_percent(self):
        """C3 strict: ice_plate error < 5%."""
        results = run_ice_plate_suite()
        for c in results:
            assert c["error_pct"] < 5.0, f"ice_plate {c['condition']}: {c['error_pct']:.4f}% >= 5%"
