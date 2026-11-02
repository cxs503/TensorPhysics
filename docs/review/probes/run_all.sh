#!/bin/sh
# Reproduce every R1/R3 review finding from the repository root.
# Usage:  sh docs/review/probes/run_all.sh
set -e
cd "$(dirname "$0")/../../.."
python -m pytest tests/test_physics_contracts.py tests/test_platform_reporting.py -q
python docs/review/probes/probe_r1r3_runner_contracts.py
python docs/review/probes/probe_registry_url_leak.py
python docs/review/probes/probe_gate_bypass_and_event_loop.py
python docs/review/probes/make_figures.py
