# DEM3D fixed-duration time-step refinement campaign

After the TensorDEM solver exposes `--dt-s` and `--duration-s`, run:

```bash
python examples/research_campaign_dem3d_timestep_refinement.py \
  --tensordem-root ../TensorDEM \
  --work-dir outputs/dem3d-timestep-refinement \
  --peak-load-N 0.02 \
  --base-steps 24 \
  --relative-tolerance 0.05
```

The controller first runs the baseline at the solver's conservative default time step,
then repeats at half and quarter of that time step. All three runs use the same peak
load and simulated physical duration; the step count is increased as the time step is
refined. The report gate independently verifies solver protocol/verdict, exact load,
positive time-step metadata, duration = dt × steps, history length, finite histories,
monotonic time and fracture counts, and valid signatures.

Artifacts:
- `timestep_refinement_report.json`: per-level observables, adjacent-level relative
  differences, declared tolerance, independent evidence gates, and campaign signature.
- `timestep_refinement_metrics.csv`: machine-readable per-level response values.

The default 5% relative tolerance is an explicit exploratory criterion, not a universal
scientific standard. A failed tolerance yields `not_within_declared_tolerance` while
the run may still have passed its numerical evidence gates. The campaign does not
force a monotonic response or require broken-bond counts to match; fracture is discrete
and the count is diagnostic. A tolerance pass is not proof of mathematical convergence,
physical validity, or agreement with experiments.

This is a **time-step sensitivity** study only. It does not change particle resolution.
A separate resolution study must define geometrically comparable models and account
for how particle radius, bond network, material stiffness and total specimen geometry
are scaled before comparing outcomes.
