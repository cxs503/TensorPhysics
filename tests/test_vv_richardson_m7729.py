"""Tests for the high-level Richardson extrapolation / GCI engine.

Positive controls: an exact manufactured solution ``f(h) = f_exact + C h**p``
on three grids must return the correct ``p``, ``f_ext`` to round-off and a
GCI that decreases under refinement.  Negative controls: non-asymptotic
sequences (oscillatory, diverging) must never return a silent PASS.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.vv import (
    ConvergenceStatus,
    GCIResult,
    RichardsonExtrapolation,
)


class TestRichardsonBasic:
    """Basic Richardson extrapolation tests."""

    def test_perfect_second_order_convergence(self):
        """Exact second-order convergence (p=2)."""
        # f_exact = 1.0, f_h = f_exact + C*h^2
        # h_coarse=4h, h_medium=2h, h_fine=h
        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16  # h=4
        f_medium = f_exact + C * 4   # h=2
        f_fine = f_exact + C * 1     # h=1

        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)

        assert abs(result.p - 2.0) < 1e-10, f"Expected p=2, got p={result.p}"
        assert abs(result.f_ext - f_exact) < 1e-10, f"Expected f_ext={f_exact}, got {result.f_ext}"
        assert result.status == ConvergenceStatus.PASS

    def test_perfect_first_order_convergence(self):
        """Exact first-order convergence (p=1)."""
        C = 0.1
        f_exact = 2.0
        f_coarse = f_exact + C * 4  # h=4
        f_medium = f_exact + C * 2  # h=2
        f_fine = f_exact + C * 1    # h=1

        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)

        assert abs(result.p - 1.0) < 1e-10, f"Expected p=1, got p={result.p}"
        assert abs(result.f_ext - f_exact) < 1e-10

    def test_third_order_convergence(self):
        """Exact third-order convergence (p=3)."""
        C = 0.001
        f_exact = 5.0
        f_coarse = f_exact + C * 64  # h=4, h^3=64
        f_medium = f_exact + C * 8   # h=2, h^3=8
        f_fine = f_exact + C * 1     # h=1, h^3=1

        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)

        assert abs(result.p - 3.0) < 1e-10, f"Expected p=3, got p={result.p}"
        assert abs(result.f_ext - f_exact) < 1e-10

    def test_gci_decreases_with_refinement(self):
        """GCI_fine should be smaller than GCI_coarse."""
        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1

        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)

        assert result.gci_fine < result.gci_coarse
        assert result.gci_fine < 5.0  # Should be small for well-converged

    def test_gci_ratio_near_one(self):
        """GCI ratio should be ~1.0 in the asymptotic range."""
        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1

        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)

        assert abs(result.gci_ratio - 1.0) < 0.5


class TestRichardsonNegative:
    """Negative controls: non-asymptotic sequences must not PASS."""

    def test_oscillatory_convergence(self):
        """Oscillatory sequence (sign changes) must FAIL."""
        # f values oscillate: 1.0, 1.1, 1.0
        re = RichardsonExtrapolation()
        result = re.compute(1.0, 1.1, 1.0, r_fine=2.0, r_coarse=2.0)

        assert result.status == ConvergenceStatus.FAIL
        assert not result.asymptotic

    def test_diverging_sequence(self):
        """Diverging sequence must FAIL."""
        # f values diverge: 1.0, 2.0, 4.0
        re = RichardsonExtrapolation()
        result = re.compute(4.0, 2.0, 1.0, r_fine=2.0, r_coarse=2.0)

        assert result.status == ConvergenceStatus.FAIL

    def test_invalid_refinement_ratio(self):
        """Refinement ratio must be > 1."""
        re = RichardsonExtrapolation()
        with pytest.raises(ValueError):
            re.compute(1.0, 1.1, 1.2, r_fine=1.0, r_coarse=2.0)

    def test_invalid_safety_factor(self):
        """Safety factor must be positive."""
        with pytest.raises(ValueError):
            RichardsonExtrapolation(safety_factor=0.0)

    def test_invalid_gci_threshold(self):
        """GCI threshold must be positive."""
        with pytest.raises(ValueError):
            RichardsonExtrapolation(gci_threshold=0.0)


class TestGCIResult:
    """GCIResult serialisation tests."""

    def test_to_dict(self):
        """GCIResult should serialise to a JSON-friendly dict."""
        re = RichardsonExtrapolation()
        result = re.compute(1.01, 1.04, 1.16, r_fine=2.0, r_coarse=2.0)

        d = result.to_dict()
        assert d["p"] == result.p
        assert d["f_ext"] == result.f_ext
        assert d["gci_fine"] == result.gci_fine
        assert d["status"] == result.status.value
        assert d["asymptotic"] == result.asymptotic
        assert set(d.keys()) == {
            "f_coarse", "f_medium", "f_fine", "r_coarse", "r_fine",
            "p", "f_ext", "epsilon_fine", "epsilon_coarse",
            "gci_fine", "gci_coarse", "gci_ratio", "status", "asymptotic",
        }

    def test_is_gci_result(self):
        """compute() should return a GCIResult."""
        re = RichardsonExtrapolation()
        result = re.compute(1.01, 1.04, 1.16, r_fine=2.0, r_coarse=2.0)
        assert isinstance(result, GCIResult)
