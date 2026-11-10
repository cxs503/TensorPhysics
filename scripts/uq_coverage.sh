#!/usr/bin/env bash
# Coverage measurement for the uncertainty-budget package.
#
# The suite is split into chunks so that each process stays well under the
# 2-minute shell budget of the control plane; the runs are appended into one
# coverage data file, then exported to JSON for the verification artefact.
#
# Usage:  bash scripts/uq_coverage.sh
# Output: artifacts/uq_maturity/coverage.json
set -euo pipefail

cd "$(dirname "$0")/.."
rm -f .coverage .coverage.*

run() {  # run <first-run?> <test files...>
  local first="$1"; shift
  local flag="-a"
  [ "$first" = "first" ] && flag=""
  python -m coverage run $flag --source=src/tensorphysics/uq -m pytest "$@" -q
}

run first tests/test_uq_holton.py tests/test_uq_failsafe.py
run append tests/test_uq_solver_outputs.py tests/test_uq_cli.py tests/test_uq_gate_matrix.py
run append tests/test_uq_pce.py tests/test_uq_verdict.py
run append tests/test_uq_workflow.py tests/test_uq_gate.py
run append tests/test_uq_sample_spec.py tests/test_uq_sampler.py tests/test_uq_sobol_mc.py tests/test_uq_refinement.py
run append tests/test_uq_adapter.py tests/test_uq_ci_coverage.py
run append tests/test_uq_shift_invariance.py tests/test_uq_shard_worker.py
run append tests/test_uq_pce_sobol_gate.py

mkdir -p artifacts/uq_maturity
python -m coverage json -o artifacts/uq_maturity/coverage.json
python -m coverage report --include="src/tensorphysics/uq/*"