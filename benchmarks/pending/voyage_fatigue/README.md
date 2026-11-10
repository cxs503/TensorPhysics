# Voyage-accumulated ice-induced fatigue (mission m9281)

Single-window fatigue (m7698) is lifted to a whole voyage and a whole service
life: `D = sum_i n_i/N_i` over every leg of the route, split into
quasi-static beam response and ice-induced hull-girder vibration, plus a
linear-Miner residual-life prediction (`N_remaining = (1 - D)/D_voyage`).

## Files

| file | content |
|---|---|
| `run_voyage_fatigue.py` | runs the reference case, evaluates criteria C1–C8, writes `results.json` and 7 figures |
| `make_report.py` | renders `docs/VOYAGE_ACCUMULATED_FATIGUE.md` from `results.json` (no hand-typed numbers) |
| `results.json` | every acceptance number + provenance (git rev, python/numpy, figure sha256) |
| `figures/` | stress decomposition, rainflow spectrum, per-leg damage split, D(t) trajectory, f/f1 regime map, S-N fit, dt convergence |

## Module under test

`src/tensorphysics/voyage_fatigue/voyage_fatigue.py`
(`PowerLawSN`, `damage_from_ranges`, `rainflow_counting_reference`, `HullGirder`,
`build_hull_girder_vibration`, `ice_force_window`, `leg_damage`, `voyage_damage`,
`accumulate_voyages`, `remaining_life`, `build_reference_case`).

## Reproduce

```bash
python benchmarks/pending/voyage_fatigue/run_voyage_fatigue.py
python benchmarks/pending/voyage_fatigue/make_report.py
python -m pytest tests/test_voyage_fatigue.py -q
```

Expected: all 8 criteria PASS, `D_voyage = 8.23e-4`, ice-induced vibration
share 54.47 %, `N_total = 1215` voyages (39.9 years at 12 days/voyage), 25 tests
green.

## Verification anchors (see `results.json`)

* Miner vs hand calculation: 0.0 % deviation.
* Power-law S-N fit on the DNV D steep segment: m = 3.0000, R² = 1.000000.
* Rainflow: analytic 29 cycles (0.0 %), independent four-point counter (0.0 %).
* Vibratory stress at resonance vs `F/(k 2 zeta)`: 0.16 %.
* Damage vs time step (dt 0.02 s vs 0.005 s): 0.27 %.

## Known defect fixed here

`fatigue.rainflow._find_peaks_valleys` lost extrema across plateaus (exact-zero
gaps between pulses, clipped peaks) because `diff[i] * diff[i-1] < 0` never
fires where `diff == 0`. Fixed by collapsing plateaus before the sign-change
scan (bit-identical for repeat-free signals); regression test included.
