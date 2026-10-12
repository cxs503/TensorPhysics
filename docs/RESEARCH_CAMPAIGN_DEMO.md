# Bounded two-run TensorDEM research campaign

This campaign builds on the single-run [research loop](RESEARCH_LOOP_DEMO.md).
It executes a fixed two-point design (baseline `nx`, then `nx + 1`) with the
same `ny`, step budget, and CPU execution settings. Each run is independently
validated and archived. If a run fails its integrity gate, the campaign stops
rather than spending the remaining run budget. The campaign report records the
decision and references each run's manifest artifact.

## Run

```bash
python examples/research_campaign_dem.py --campaign-id dem-campaign-001 --nx 7 --ny 3 --steps 20
```

Outputs live under `outputs/research-campaign-dem/`, with each run in its own
round directory and a top-level `campaign_report.json`.

The parameter change is intentionally simple so integration behavior is
reproducible. It is not an endorsed scientific sensitivity design, and passing
finite-output checks is not evidence of physical accuracy. The next phase should
replace this fixed design with a benchmark-defined parameter policy and add
optional remote artifact/registry integration using shared storage.
