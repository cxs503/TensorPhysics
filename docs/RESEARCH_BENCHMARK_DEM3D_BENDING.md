# DEM3D bending research loop

This example connects the existing TensorDEM three-point-bending numerical fixture
to TensorPhysics research contracts and local artifact provenance.

## Run

Check out TensorPhysics and TensorDEM side by side, install both projects, then run:

```bash
python examples/research_benchmark_dem3d_bending.py \
  --tensordem-root ../TensorDEM \
  --work-dir outputs/dem3d-bending \
  --run-id dem3d-bending-001 \
  --steps 24
```

The controller invokes TensorDEM's
`scripts/benchmark_dem3d_three_point_bending.py`, independently reads its JSON
report and CSV history, checks finite numeric values and monotonic time/fracture
history, stores the history, benchmark report and validation report in
`LocalArtifactStore`, verifies each stored artifact, and writes a run manifest
and research-memory record. A failed gate exits non-zero.

## Evidence and limitations

- `demo_result.json`: compact controller result and explicit scientific status.
- `solver-output/three_point_bending_history.csv`: load/response history.
- `solver-output/three_point_bending_report.json`: TensorDEM fixture report.
- `validation_report.json`: independent post-run checks.
- `run_manifest.json`: configuration, revisions, output references and gate status.
- `research_memory.json`: evidence URIs and next recommended action.
- `artifact-store/`: checksum-verifiable local evidence copies.

**A passing run is only a numerical load-transfer/fracture-tracking smoke fixture.**
The existing benchmark uses fixed support particles and distributed nodal loading;
it does not establish calibrated ice flexural strength or experimental agreement.
The next scientific step is to implement/verify physical support and loading
conditions, calibrate against independent material experiments, and predeclare
quantitative acceptance thresholds before claiming physical validation.
