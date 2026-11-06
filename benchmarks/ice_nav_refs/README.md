# `benchmarks/ice_nav_refs` — sha256-anchored reference-value corpus for ice navigation

Reference data for the ice-navigation performance chain, frozen as
content-addressed assets so that campaign nodes can run **reference validation**
instead of re-typing literature numbers into tests.

```
benchmarks/ice_nav_refs/
├── build_corpus.py            # deterministic builder: every number is computed here
├── run_acceptance.py          # machine acceptance run -> artifacts/ice_nav_refs/
├── MANIFEST.json              # sha256 of every asset file + aggregate digest
└── assets/
    ├── lindqvist_1989.json                 (4 assets)
    ├── keinonen_brash.json                 (3 assets)
    ├── riska_kujala_pressure_area.json     (3 assets)
    ├── iacs_ur_i_pc6.json                  (3 assets)
    └── polar_code_i_a.json                 (3 assets)
```

16 assets / 5 families; 11 loader-level negative probes (no-citation entries and friends) + a byte-tamper probe are refused by the loader. Rebuild: `python benchmarks/ice_nav_refs/build_corpus.py`;
verify only: `... build_corpus.py --check`; acceptance: `python benchmarks/ice_nav_refs/run_acceptance.py`.

## Asset schema

| field | meaning |
|---|---|
| `asset_id` | stable identifier, unique across the corpus |
| `source_citation` | full citation (author, year, venue, pages/clause). **Mandatory**: a missing, empty, placeholder (`n/a`, `TBD`, …) or digit-free citation is rejected by the loader |
| `values` | mapping `name -> number` (or `text` for categorical clauses); names carry the unit suffix (`p_A1_MPa`, `R_c_N`, `h_nominal_m`) |
| `tolerance` | `{mode: relative\|absolute\|band\|exact, value, lower?, upper?, note}` |
| `applicable_regime` | applicability box: `ice_regime`, `ice_thickness_m: [min,max]`, plus speed/area/class ranges |
| `units` | unit string for every numeric value |
| `formula` | the expression actually evaluated (auditable, re-derivable) |
| `evidence_tier` | `published_formula` · `repo_transcription` · `derived_from_rule_form` · `literature_band` · `standards_clause` |
| `role` | `accuracy_reference` (may judge a computation) · `regression_anchor` (drift detection only — **never** quote as external truth) |
| `primary_source_verified` | `true` only if the value was checked against the primary document |
| `verification_note` | how the number was obtained and what is *not* established |
| `value_sha256` | sha256 over `{asset_id, values, tolerance}` — pins the numbers independently of the file |

Tolerances are deliberately tiered: `1e-3` for a closed-form evaluation
(arithmetic only), `0.02` for a transcribed coefficient, `0.15–0.30` for rule-form
pressures and brash-ice coefficients (band width of the underlying scatter).

## Consumption

```python
from tensorphysics.ice_nav_refs import load_corpus

corpus = load_corpus()                                  # verifies every sha256 anchor
verdict = corpus.check("RA_LOCAL_PRESSURE_TABLE", {"p_A1_MPa": 3.02})
assert verdict["pass"]                                  # within +-20 %
batch = corpus.certify({"LIN_C_CRUSH_PHI30": {"C_c": 2.1558}})      # audit block
anchors = corpus.value_anchors()                        # {asset_id: sha256}
```

`load_corpus` raises `CorpusIntegrityError` on any byte-level drift, and
`MissingCitationError` / `MissingFieldError` / `ToleranceError` on a malformed
asset — a corpus entry without provenance cannot enter the loader at all.

## Reading discipline

* `role = regression_anchor` assets freeze *this repository's* transcription.
  They detect drift; they are **not** validation evidence. `LIN_RESISTANCE_COMPONENTS`
  in particular inherits a known breadth-factor defect (see
  `docs/ICE_NAV_REFS_CORPUS.md`, §"Known defects caught by this corpus").
* `primary_source_verified = false` means the standard/monograph itself was not
  accessible in the build environment. Consume the `verification_note` before
  upgrading any such asset in an evidence table.
* The corpus is validation data, not a ship-design feature: no resistance,
  propulsion or scantling kernels live here.