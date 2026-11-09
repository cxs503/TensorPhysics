"""Tests for the UQ PCE Sobol' gate end-to-end and report scripts.

These tests verify that the matured mission deliverables (documentation,
end-to-end script, report generation) work correctly and produce valid outputs.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"


def _run_script(script: str, *args: str, cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    """Run a script and return the result."""
    env = {"PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        env={**subprocess.os.environ, **env},
        timeout=300,
    )


class TestUQPceSobolGateE2E:
    """Tests for the end-to-end demonstration script."""

    def test_script_runs_successfully(self, tmp_path: pathlib.Path) -> None:
        """The e2e script should run without errors."""
        result = _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    def test_produces_ishigami_study(self, tmp_path: pathlib.Path) -> None:
        """The script should produce ishigami_study.json."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        study_path = tmp_path / "ishigami_study.json"
        assert study_path.exists(), "ishigami_study.json not created"

        study = json.loads(study_path.read_text())
        assert study["benchmark"] == "ishigami"
        assert study["status"] in {"PASS", "FAIL", "WITHHELD"}
        assert "indices" in study
        assert "surrogate" in study

    def test_produces_g_function_study(self, tmp_path: pathlib.Path) -> None:
        """The script should produce g_function_study.json."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        study_path = tmp_path / "g_function_study.json"
        assert study_path.exists(), "g_function_study.json not created"

        study = json.loads(study_path.read_text())
        assert study["benchmark"] == "sobol_g"
        assert study["status"] in {"PASS", "FAIL", "WITHHELD"}
        assert "indices" in study

    def test_ishigami_indices_are_accurate(self, tmp_path: pathlib.Path) -> None:
        """Ishigami PCE indices should match analytic values within 1e-3."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        study = json.loads((tmp_path / "ishigami_study.json").read_text())

        # Check first-order indices
        for name in ["x1", "x2", "x3"]:
            data = study["indices"][name]
            assert abs(data["pce"] - data["analytic"]) <= 1e-3, (
                f"{name} first-order error too large: {abs(data['pce'] - data['analytic'])}"
            )

        # Check total-effect indices
        for name in ["x1", "x2", "x3"]:
            data = study["indices"][f"{name}_total"]
            assert abs(data["pce"] - data["analytic"]) <= 1e-3, (
                f"{name} total-effect error too large: {abs(data['pce'] - data['analytic'])}"
            )

    def test_g_function_indices_are_accurate(self, tmp_path: pathlib.Path) -> None:
        """G-function PCE indices should match analytic values within 5e-3."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        study = json.loads((tmp_path / "g_function_study.json").read_text())

        for name in ["x1", "x2", "x3", "x4"]:
            data = study["indices"][name]
            assert abs(data["pce"] - data["analytic"]) <= 5e-3, (
                f"{name} first-order error too large: {abs(data['pce'] - data['analytic'])}"
            )

    def test_surrogate_quality_is_good(self, tmp_path: pathlib.Path) -> None:
        """Surrogate ε_LOO should be below 0.1."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))

        for study_name in ["ishigami_study.json", "g_function_study.json"]:
            study = json.loads((tmp_path / study_name).read_text())
            eps_loo = study["surrogate"]["eps_loo"]
            assert eps_loo <= 0.1, f"{study_name} ε_LOO too large: {eps_loo}"
            assert study["surrogate"]["r2"] > 0.99, f"{study_name} R² too low"

    def test_acceptance_criteria_pass(self, tmp_path: pathlib.Path) -> None:
        """All acceptance criteria should pass."""
        result = _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        assert result.returncode == 0, "Acceptance criteria failed"

        # Verify from JSON
        for study_name in ["ishigami_study.json", "g_function_study.json"]:
            study = json.loads((tmp_path / study_name).read_text())
            for crit_name, crit in study["acceptance"].items():
                assert crit["pass"], f"{study_name} {crit_name} failed"


class TestUQPceSobolGateReport:
    """Tests for the report generation script."""

    def test_script_runs_successfully(self, tmp_path: pathlib.Path) -> None:
        """The report script should run without errors."""
        # First generate a study
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))

        # Then generate report
        report_path = tmp_path / "report.md"
        result = _run_script(
            "uq_pce_sobol_gate_report.py",
            "--study", str(tmp_path / "ishigami_study.json"),
            "--out", str(report_path),
        )
        assert result.returncode == 0, f"Report script failed:\n{result.stderr}"
        assert report_path.exists()

    def test_report_contains_required_sections(self, tmp_path: pathlib.Path) -> None:
        """The report should contain all required sections."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        report_path = tmp_path / "report.md"
        _run_script(
            "uq_pce_sobol_gate_report.py",
            "--study", str(tmp_path / "ishigami_study.json"),
            "--out", str(report_path),
        )

        report = report_path.read_text()
        assert "# UQ PCE Sobol' Gate Report" in report
        assert "## 1. Input Specification" in report
        assert "## 2. Surrogate Quality" in report
        assert "## 3. Sobol' Indices" in report
        assert "## 5. Verdict" in report
        assert "## References" in report

    def test_report_numbers_match_study(self, tmp_path: pathlib.Path) -> None:
        """Report numbers should match the study JSON (no hand-copied digits)."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))

        study = json.loads((tmp_path / "ishigami_study.json").read_text())
        report_path = tmp_path / "report.md"
        _run_script(
            "uq_pce_sobol_gate_report.py",
            "--study", str(tmp_path / "ishigami_study.json"),
            "--out", str(report_path),
        )

        report = report_path.read_text()

        # Check that key numbers appear in the report
        eps_loo = study["surrogate"]["eps_loo"]
        assert f"{eps_loo:.4f}" in report, "ε_LOO not in report"

        # Check that status appears
        assert study["status"] in report, "Status not in report"

    def test_report_handles_missing_mc(self, tmp_path: pathlib.Path) -> None:
        """Report should handle studies without Monte Carlo gracefully."""
        _run_script("uq_pce_sobol_gate_e2e.py", "--out", str(tmp_path))
        report_path = tmp_path / "report_g.md"
        result = _run_script(
            "uq_pce_sobol_gate_report.py",
            "--study", str(tmp_path / "g_function_study.json"),
            "--out", str(report_path),
        )
        assert result.returncode == 0
        report = report_path.read_text()
        assert "Cross-check not available" in report or "## 4." in report


class TestDocumentation:
    """Tests for the documentation file."""

    def test_documentation_exists(self) -> None:
        """The UQ_PCE_SOBOL_GATE.md documentation should exist."""
        doc_path = ROOT / "docs" / "UQ_PCE_SOBOL_GATE.md"
        assert doc_path.exists(), "Documentation file not found"

    def test_documentation_has_required_sections(self) -> None:
        """The documentation should cover all key topics."""
        doc_path = ROOT / "docs" / "UQ_PCE_SOBOL_GATE.md"
        doc = doc_path.read_text()

        required_sections = [
            "## 1. Overview",
            "## 2. Mathematical Foundation",
            "## 3. Pipeline",
            "## 4. Usage",
            "## 5. Acceptance Criteria",
            "## 6. Verdict System",
            "## 7. Limitations",
            "## 8. References",
        ]

        for section in required_sections:
            assert section in doc, f"Missing section: {section}"

    def test_documentation_includes_code_examples(self) -> None:
        """The documentation should include usage examples."""
        doc_path = ROOT / "docs" / "UQ_PCE_SOBOL_GATE.md"
        doc = doc_path.read_text()

        assert "```python" in doc, "No Python code examples"
        assert "run_uq_study" in doc, "run_uq_study not documented"
        assert "ishigami" in doc, "Ishigami benchmark not mentioned"
