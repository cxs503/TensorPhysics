# TensorPhysics

TensorPhysics provides solver-neutral scientific workflow primitives, coupling interfaces, and provenance-aware execution helpers.

## First executable research-loop demo

The first bounded TensorDEM integration slice is documented in
[Research Loop Demo](docs/RESEARCH_LOOP_DEMO.md) and implemented in
`examples/research_loop_dem.py`. It runs a real solver process, independently
checks history/trajectory readability and finite values, stores outputs and
validation evidence through `LocalArtifactStore`, emits a typed `RunManifest`,
and recommends a next action based on the checks. Versioned goal, plan,
validation-report, and research-memory contracts live in
`tensorphysics.research_contracts`.

This is a software-integrity gate, not physical validation; it does not yet
register the run in TensorPlatform or perform an AI-driven parameter search.

## Bounded multi-run campaign

[Research Campaign Demo](docs/RESEARCH_CAMPAIGN_DEMO.md) adds a two-run, fail-closed campaign controller around the single-run demo. It varies one input parameter, independently validates each real TensorDEM run, records per-run manifest references, and stops on a failed gate. CI executes both runs. This remains a software-integrity integration exercise, not a physical sensitivity study or AI-driven research policy.
