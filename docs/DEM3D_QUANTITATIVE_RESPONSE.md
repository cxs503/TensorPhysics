# DEM3D quantitative response metrics

Run after a successful load-amplitude campaign:

```bash
python examples/analyze_dem3d_load_sensitivity.py \
  --work-dir outputs/dem3d-load-sensitivity
```

The analyzer reopens the solver report from each `run-NN/` directory, checks the
solver protocol, requested/final load identity, signature format and finite ordered
history, then emits:

- `quantitative_response_report.json`: per-run values, metric definitions,
  scientific limitations, and a deterministic SHA-256 signature over the ordered
  campaign evidence.
- `response_metrics.csv`: one row per load case, suitable for plotting or ingestion.

## Definitions

- **Load-transfer ratio** = absolute final support reaction / requested peak load.
- **Relative force-balance residual** = absolute final force-balance residual / requested peak load.
- **Secant compliance** = absolute final midspan deflection / requested peak load.
- **Secant stiffness** = requested peak load / absolute final midspan deflection
  (null if deflection is zero).
- **Relative changes** = fractional change from the preceding load case for
  compliance and load-transfer ratio (null if the baseline metric is zero).

These quantities summarize the current discrete model's numerical response and
are not acceptance thresholds. The analyzer does not force a presumed monotonic
response trend. None of these checks establishes calibrated material properties,
experimental agreement, physical validity, or numerical convergence. The fixed
particle supports and distributed nodal loading remain fixture choices that need
independent physical justification.
