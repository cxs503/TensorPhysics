"""Cross-solver adjudication: unified framework for code-to-code comparison.

This module orchestrates the complete cross-solver adjudication process:

1. Lid-driven cavity: LBM/FVM/FEM vs Ghia (Re=100, 400)
2. MMS convergence: DG vs FVM on Kovasznay flow
3. SPH↔LBM anchors: Poiseuille + shallow water
4. Error attribution: 24.9% residual decomposition

Produces a comprehensive error budget JSON and supports one-click re-computation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .lid_cavity import run_lid_cavity_comparison
from .mms_convergence import run_mms_convergence_study
from .sph_lbm_anchors import run_sph_lbm_anchors
from .error_attribution import run_24_9_attribution

__all__ = [
    "CrossSolverAdjudicator",
    "AdjudicationResult",
    "run_full_adjudication",
]


@dataclass
class AdjudicationResult:
    """Complete adjudication result with all comparisons."""
    lid_cavity_re100: dict
    lid_cavity_re400: dict
    mms_convergence: dict
    sph_lbm_anchors: dict
    error_attribution: dict
    timestamp: str
    summary: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lid_cavity_re100": self.lid_cavity_re100,
            "lid_cavity_re400": self.lid_cavity_re400,
            "mms_convergence": self.mms_convergence,
            "sph_lbm_anchors": self.sph_lbm_anchors,
            "error_attribution": self.error_attribution,
            "timestamp": self.timestamp,
            "summary": self.summary,
        }

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class CrossSolverAdjudicator:
    """Orchestrates cross-solver adjudication.

    Runs all comparison tests and produces unified error budget.
    """
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path("artifacts/cross_solver_adjudication")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_lid_cavity(self, Re: int = 100, nx: int = 65) -> dict:
        """Run lid-driven cavity comparison."""
        return run_lid_cavity_comparison(Re=Re, nx=nx, ny=nx)

    def run_mms(self, Re: float = 40.0, dg_degree: int = 2) -> dict:
        """Run MMS convergence study."""
        return run_mms_convergence_study(Re=Re, dg_degree=dg_degree)

    def run_anchors(self) -> dict:
        """Run SPH↔LBM anchor tests."""
        return run_sph_lbm_anchors()

    def run_attribution(self) -> dict:
        """Run 24.9% error attribution."""
        return run_24_9_attribution()

    def adjudicate(self) -> AdjudicationResult:
        """Run complete adjudication suite."""
        # 1. Lid-driven cavity at Re=100 and Re=400
        lc_re100 = self.run_lid_cavity(Re=100, nx=65)
        lc_re400 = self.run_lid_cavity(Re=400, nx=65)

        # 2. MMS convergence
        mms = self.run_mms(Re=40.0, dg_degree=2)

        # 3. SPH↔LBM anchors
        anchors = self.run_anchors()

        # 4. Error attribution
        attribution = self.run_attribution()

        # Summary
        summary = {
            "lid_cavity_re100_status": lc_re100["status"],
            "lid_cavity_re400_status": lc_re400["status"],
            "mms_status": mms["status"],
            "anchors_status": anchors["status"],
            "attribution_status": attribution["status"],
            "overall_status": "PASS" if all([
                lc_re100["status"] == "PASS",
                lc_re400["status"] == "PASS",
                mms["status"] == "PASS",
                anchors["status"] == "PASS",
                attribution["status"] == "PASS",
            ]) else "FAIL",
            "total_comparisons": 5,
            "passed_comparisons": sum([
                lc_re100["status"] == "PASS",
                lc_re400["status"] == "PASS",
                mms["status"] == "PASS",
                anchors["status"] == "PASS",
                attribution["status"] == "PASS",
            ]),
        }

        result = AdjudicationResult(
            lid_cavity_re100=lc_re100,
            lid_cavity_re400=lc_re400,
            mms_convergence=mms,
            sph_lbm_anchors=anchors,
            error_attribution=attribution,
            timestamp=datetime.now().isoformat(),
            summary=summary,
        )

        # Save
        result.save(self.output_dir / "adjudication_result.json")

        return result


def run_full_adjudication(output_dir: Optional[Path] = None) -> AdjudicationResult:
    """Run complete cross-solver adjudication.

    Parameters
    ----------
    output_dir : Path, optional
        Directory for output files.

    Returns
    -------
    AdjudicationResult with all comparisons.
    """
    adjudicator = CrossSolverAdjudicator(output_dir=output_dir)
    return adjudicator.adjudicate()
