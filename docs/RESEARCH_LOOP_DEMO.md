# First closed-loop demo: TensorPhysics + TensorDEM

This demo is a first executable integration slice, not a validated icebreaking
simulation. It runs one bounded TensorDEM case, checks outputs independently,
stores artifacts with SHA-256/size verification, writes a typed run manifest,
and emits a deterministic next-action record.

## Install

In an environment where TensorDEM and TensorPhysics are installed:

```bash
python -m pip install -e ".[test]"
python -m pip install "git+https://github.com/cxs503/TensorDEM.git"
```

## Run

```bash
python examples/research_loop_dem.py --run-id research-loop-dem-001 --nx 7 --ny 3 --steps 20
```

Outputs are written under `outputs/research-loop-dem/`:

- `solver-output/history.csv` and `solver-output/trajectory.pt`: real solver outputs.
- `validation_report.json`: independent finite-value, readability, and time-order checks.
- `run_manifest.json`: goal/plan provenance, code revisions, artifact references, and validation state.
- `artifact-store/<run-id>/...`: checksum-verified local artifact copies.
- `research_memory.json`: evidence references and next action.
- `demo_result.json`: combined machine-readable run summary.

The next action is `run_bounded_parameter_sweep` only when the checks pass;
otherwise the loop recommends `stop_and_repair_before_retry`. This is a
transparent deterministic policy, not an LLM-driven autonomous research agent.

## What this proves and does not prove

It tests a real TensorDEM invocation, finite/readable outputs, local artifact
integrity, and traceable typed records. It does **not** prove physical correctness,
benchmark agreement, convergence, fracture-law validity, or ship resistance
accuracy. It does not upload artifacts to TensorPlatform or S3. Platform
registration and a scientifically meaningful benchmark gate are later steps.
