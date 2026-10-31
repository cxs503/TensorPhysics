#!/usr/bin/env python3
"""Submit the Hetényi anchor report to the Tensor API."""

import json
import os
import sys
import base64
from pathlib import Path

import requests

# Get environment variables
API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USERNAME = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")

def get_token() -> str:
    """Get authentication token."""
    url = f"{API_BASE}/auth/login"
    data = {"username": USERNAME, "password": PASSWORD}
    try:
        resp = requests.post(url, json=data, timeout=10)
        resp.raise_for_status()
        return resp.json().get("access_token", "")
    except Exception as e:
        print(f"Warning: Could not get token: {e}")
        return ""

def fig_to_base64(path: Path) -> str:
    """Convert PNG file to base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    """Submit the report."""
    artifacts_dir = Path(__file__).resolve().parent
    
    # Load report data
    report_path = artifacts_dir / "hetenyi_anchor_report.json"
    with open(report_path) as f:
        report_data = json.load(f)
    
    # Load figures
    fig1_path = artifacts_dir / "fig_convergence.png"
    fig2_path = artifacts_dir / "fig_deflection_field.png"
    
    figures = []
    if fig1_path.exists():
        figures.append({
            "name": "fig_convergence",
            "caption": "Hetényi series convergence: (a) center deflection vs. n_terms, (b) relative truncation error vs. n_terms. The 100-term truncation achieves 10⁻¹¹ relative error.",
            "png_base64": fig_to_base64(fig1_path),
        })
    if fig2_path.exists():
        figures.append({
            "name": "fig_deflection_field",
            "caption": "Deflection field w(x,y): (a) contour plot showing simply-supported BCs (w=0 on edges), (b) centerline profiles confirming four-fold symmetry.",
            "png_base64": fig_to_base64(fig2_path),
        })
    
    # Build report payload
    report_payload = {
        "title": "Hetényi (1946) Series Solution Anchor – Simply-Supported Square Plate on Winkler Foundation",
        "summary": "Validated Hetényi double Fourier series for SS square plate on Winkler foundation. All 3 acceptance criteria PASS: truncation_100_error=1.01e-11 (<0.01), convergence_monotone=True, pytest_anchor=PASS. Reference solution for FEM ice-plate validation.",
        "scenario_key": "SIR-PC2|V01",
        "mission_id": 721,
        "tags": ["validation", "ice-sheet", "hetenyi", "winkler", "plate", "fourier-series", "benchmark"],
        "markdown": open(Path(__file__).resolve().parent.parent / "docs" / "HETENYI_ANCHOR_REPORT.md").read(),
        "figures": figures,
    }
    
    # Get token
    token = get_token()
    if not token:
        print("Warning: No token obtained. Saving report locally only.")
        # Save locally
        output_path = artifacts_dir / "report_submission.json"
        with open(output_path, "w") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Report saved locally: {output_path}")
        return
    
    # Submit report
    url = f"{API_BASE}/tensor/reports"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    try:
        resp = requests.post(url, json=report_payload, headers=headers, timeout=30)
        resp.raise_for_status()
        print(f"Report submitted successfully: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Error submitting report: {e}")
        # Save locally as fallback
        output_path = artifacts_dir / "report_submission.json"
        with open(output_path, "w") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Report saved locally: {output_path}")

if __name__ == "__main__":
    main()
