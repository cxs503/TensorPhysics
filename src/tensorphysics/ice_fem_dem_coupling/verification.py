"""Full verification runner for the FEM-DEM ice-structure coupling campaign.

This module runs all verification criteria (C1–C5) and produces a comprehensive
report with pass/fail status for each criterion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .plate_bending import solve_plate_bending_verification, PlateBendingResult
from .fracture_handoff import run_fracture_handoff, FractureHandoffResult
from .floe_drift import (
    simulate_floe_drift,
    FloeProperties,
    DriftEnvironment,
    FloeDriftResult,
    MT_UIKKU_REFERENCE,
)
from .ice_load_history import run_ice_load_verification, IceLoadResult
from .energy_audit import run_energy_audit, EnergyAuditResult


@dataclass
class VerificationReport:
    """Comprehensive verification report for p299w4-n04."""
    timestamp: str
    # Individual criteria
    c1_plate_bending: Optional[PlateBendingResult] = None
    c2_fracture_handoff: Optional[FractureHandoffResult] = None
    c3_floe_drift: Optional[FloeDriftResult] = None
    c4_ice_load: Optional[IceLoadResult] = None
    c5_energy_audit: Optional[EnergyAuditResult] = None
    # Summary
    c1_pass: bool = False
    c2_pass: bool = False
    c3_pass: bool = False
    c4_pass: bool = False
    c5_pass: bool = False
    all_pass: bool = False
    # Metadata
    n_tests: int = 0
    n_pass: int = 0

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary."""
        return {
            "timestamp": self.timestamp,
            "criteria": {
                "C1_plate_bending": {
                    "pass": self.c1_pass,
                    "stress_error_rel": self.c1_plate_bending.stress_error_rel if self.c1_plate_bending else None,
                    "threshold": 0.05,
                },
                "C2_fracture_handoff": {
                    "pass": self.c2_pass,
                    "mass_error_rel": self.c2_fracture_handoff.mass_error_rel if self.c2_fracture_handoff else None,
                    "threshold": 0.01,
                },
                "C3_floe_drift": {
                    "pass": self.c3_pass,
                    "area_error_rel": self.c3_floe_drift.area_error_rel if self.c3_floe_drift else None,
                    "threshold": 0.20,
                },
                "C4_ice_load": {
                    "pass": self.c4_pass,
                    "peak_force_error_rel": self.c4_ice_load.peak_force_error_rel if self.c4_ice_load else None,
                    "weibull_k_error_rel": self.c4_ice_load.weibull_k_error_rel if self.c4_ice_load else None,
                    "threshold_peak": 0.20,
                    "threshold_weibull": 0.15,
                },
                "C5_energy_audit": {
                    "pass": self.c5_pass,
                    "energy_error_rel": self.c5_energy_audit.energy_error_rel if self.c5_energy_audit else None,
                    "threshold": 0.10,
                },
            },
            "all_pass": self.all_pass,
            "n_tests": self.n_tests,
            "n_pass": self.n_pass,
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize the report to JSON."""
        data = self.summary()
        # Add detailed results
        if self.c1_plate_bending:
            data["C1_details"] = self.c1_plate_bending.as_dict()
        if self.c2_fracture_handoff:
            data["C2_details"] = self.c2_fracture_handoff.as_dict()
        if self.c3_floe_drift:
            data["C3_details"] = self.c3_floe_drift.as_dict()
        if self.c4_ice_load:
            data["C4_details"] = self.c4_ice_load.as_dict()
        if self.c5_energy_audit:
            data["C5_details"] = self.c5_energy_audit.as_dict()

        json_str = json.dumps(data, indent=2, default=str)
        if path:
            path.write_text(json_str)
        return json_str


def run_full_verification(
    # C1: Plate bending
    P_load: float = 50e3,
    E_ice: float = 5.0e9,
    h_ice: float = 1.0,
    nu: float = 0.3,
    k_foundation: float = 9810.0,
    n_el_plate: int = 200,
    # C2: Fracture handoff
    handoff_dims: tuple = (32, 12, 4),
    # C3: Floe drift
    n_floes: int = 50,
    current_speed_m_s: float = 0.15,
    drift_duration_s: float = 3600.0,
    channel_width_m: float = 50.0,
    # C4: Ice load
    load_duration_s: float = 1000.0,
    event_rate_hz: float = 0.5,
    peak_weibull_k: float = 1.8,
    peak_weibull_lambda_N: float = 50e3,
    # C5: Energy audit
    sigma_f_Pa: float = 1.0e6,
    G_f_J_m2: float = 100.0,
) -> VerificationReport:
    """Run the full FEM-DEM ice-structure coupling verification.

    Parameters
    ----------
    Various parameters for each verification criterion.

    Returns
    -------
    VerificationReport
    """
    report = VerificationReport(timestamp=datetime.now().isoformat())

    # --- C1: Plate bending ---
    report.c1_plate_bending = solve_plate_bending_verification(
        P=P_load,
        E=E_ice,
        h=h_ice,
        nu=nu,
        k=k_foundation,
        n_el=n_el_plate,
    )
    report.c1_pass = report.c1_plate_bending.c1_pass

    # --- C2: Fracture handoff ---
    report.c2_fracture_handoff = run_fracture_handoff(
        dims=handoff_dims,
    )
    report.c2_pass = report.c2_fracture_handoff.c2_pass

    # --- C3: Floe drift ---
    rng = np.random.default_rng(42)
    floes = []
    for _ in range(n_floes):
        area = rng.uniform(10, 100)  # m²
        floe = FloeProperties(
            area_m2=area,
            thickness_m=h_ice,
            position=np.array([rng.uniform(-100, 100), rng.uniform(-channel_width_m/2, channel_width_m/2)]),
            velocity=np.zeros(2),
        )
        floes.append(floe)

    env = DriftEnvironment(
        current_velocity=np.array([current_speed_m_s, 0.0]),
        wind_velocity=np.zeros(2),
    )

    report.c3_floe_drift = simulate_floe_drift(
        floes=floes,
        env=env,
        duration_s=drift_duration_s,
        dt_s=10.0,
        channel_width_m=channel_width_m,
    )
    report.c3_pass = report.c3_floe_drift.c3_pass

    # --- C4: Ice load ---
    report.c4_ice_load = run_ice_load_verification(
        duration_s=load_duration_s,
        dt_s=0.1,
        event_rate_hz=event_rate_hz,
        peak_weibull_k_true=peak_weibull_k,
        peak_weibull_lambda_N=peak_weibull_lambda_N,
        pulse_duration_s=2.0,
        seed=42,
        peak_force_ref_N=200e3,
        weibull_k_ref=1.7,
    )
    report.c4_pass = report.c4_ice_load.c4_peak_pass and report.c4_ice_load.c4_weibull_pass

    # --- C5: Energy audit ---
    report.c5_energy_audit = run_energy_audit(
        E_ice=E_ice,
        h_ice=h_ice,
        nu=nu,
        k_foundation=k_foundation,
        P_load=P_load,
        n_el=n_el_plate,
        sigma_f_Pa=sigma_f_Pa,
        G_f_J_m2=G_f_J_m2,
    )
    report.c5_pass = report.c5_energy_audit.c5_pass

    # --- Summary ---
    report.all_pass = all([
        report.c1_pass,
        report.c2_pass,
        report.c3_pass,
        report.c4_pass,
        report.c5_pass,
    ])
    report.n_tests = 5
    report.n_pass = sum([
        report.c1_pass,
        report.c2_pass,
        report.c3_pass,
        report.c4_pass,
        report.c5_pass,
    ])

    return report
