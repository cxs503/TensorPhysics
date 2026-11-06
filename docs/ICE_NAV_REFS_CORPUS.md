# Ice-navigation reference corpus (`benchmarks/ice_nav_refs`)

Status: **implemented, machine-accepted** (mission IN10-L2 / #4907).
Machine evidence: `artifacts/ice_nav_refs/acceptance.json`,
`artifacts/ice_nav_refs/verification_matrix.json`; tests: `tests/test_refs_corpus.py`.

## 1. Why TensorPhysics

TensorPhysics owns *scientific acceptance*: the decision whether a computed
number is admissible against an external datum, and the provenance of that
datum. A reference corpus is that artefact in its most reusable form. It is not a
solver kernel (TensorLBM/FVM/FEM territory) and it is not a ship-design feature
(the domain ship platform); it is the frozen *yardstick* the ice-navigation
campaign nodes measure themselves against.

## 2. What is in the corpus

16 assets in five families, each with `source_citation`, `values`, `tolerance`
and `applicable_regime` (full schema: `benchmarks/ice_nav_refs/README.md`):

| family | assets | content |
|---|---|---|
| `lindqvist_1989` | 4 | crushing coefficient `C_c`, speed coefficient `C_s = 0.15+0.3Fr_h`, three-component resistance at the reference ship, ice/steel friction band |
| `keinonen_brash` | 3 | semi-empirical coefficient pair `(C0, C1)`, brash-ice resistance at a Baltic reference condition, coefficient band for the fragmented regime |
| `riska_kujala_pressure_area` | 3 | pressure-area exponent `ex = 0.45` (band 0.40–0.55), local pressure table `p = p0 A^-0.45` over 0.25–10 m², global pressure table `p_G = 1.2 (h/1 m)^-0.5` |
| `iacs_ur_i_pc6` | 3 | PC6 local panel pressure table over 0.5–8 m², rule ice-patch geometry (`0.6 h`, 4:1), PC6 design ice thickness band |
| `polar_code_i_a` | 3 | category A/B/C → Polar Class mapping, polar service temperature band, category thickness bands |

Reference condition: `L = 150 m`, `B = 20 m`, bow angle `30°`, `h_ice = 1.0 m`,
`V = 2 m/s`, `σ_c = 5 MPa`, `σ_f = 0.5 MPa`, `ρ_i = 900 kg/m³`, `μ = 0.15`.

## 3. Anchoring

Two independent digests:

1. **File anchor** — `MANIFEST.json` stores `sha256` and byte length of every
   `assets/*.json`, plus `aggregate_sha256 = sha256(sorted "path:sha256" lines)`.
   `tensorphysics.ice_nav_refs.verify_manifest()` recomputes all of them; a
   single tampered byte makes `load_corpus()` raise `CorpusIntegrityError`
   (probe `tampered_asset_file_byte` in `acceptance.json`).
2. **Value anchor** — `value_sha256 = sha256({asset_id, values, tolerance})`
   per asset. Rewording a citation does not move it; changing one digit does
   (probe `value_relabelled_without_redigest`). Consumers that need to pin a
   number can store the 64-hex digest without storing the JSON.

Determinism: `build_corpus.py` computes every number from the expression in the
`formula` field, so rebuilding gives byte-identical files (`rebuild_byte_identical`
probe: `aggregate_sha256` unchanged).

## 4. Negative cases (the corpus refuses to accept)

`asset_from_dict()` / `load_corpus()` reject:

| anomaly | exception |
|---|---|
| field `source_citation` absent | `MissingFieldError` |
| citation empty / `n/a` / `TBD` / `unknown` / `-` / too short / digit-free | `MissingCitationError` |
| citation not a string | `MissingCitationError` |
| `values` / `tolerance` / `applicable_regime` absent | `MissingFieldError` |
| tolerance missing `mode`/`value`, unknown mode, zero or negative, inverted band, band without bounds | `ToleranceError` |
| recorded number changed without re-digest | `CorpusIntegrityError` |
| asset file byte changed | `CorpusIntegrityError` (via manifest) |
| duplicate `asset_id` across files | `CorpusError` |
| manifest `asset_count` ≠ loaded assets | `CorpusError` |

Eleven loader-level negative probes plus a byte-tamper probe are exercised in `run_acceptance.py` and in
`tests/test_refs_corpus.py`; all are refused.

## 5. Known defects caught by this corpus

`LIN_RESISTANCE_COMPONENTS` is a `regression_anchor`, not an accuracy
reference, because the transcribed Lindqvist component set is not breadth
consistent:

* `R_c = C_c σ_c h²` has **no** breadth factor. At the reference condition it
  returns `10.78 MN`, i.e. it implies an effective crushing patch of
  `R_c/σ_c ≈ 2.16 m²` on a `20 m` beam ship — the breadth scale of the contact
  is missing from the term.
* `R_f = C_f ρ g h` uses a breadth-bearing coefficient `C_f` (units m), while
  `R_s = C_s ρ V² h B` carries `B` explicitly; the two terms therefore scale
  with breadth differently.

Consequence: `R_total ≈ 10.8 MN` at `h = 1 m` is one order of magnitude above
the `0.3–1 MN` engineering envelope for a 20 m beam in 1 m first-year ice at
2 m/s (envelope estimate, **not** a measured datum — no ice-tank series for
this hull exists in the repository). The corpus records the value so that drift
is detectable, and simultaneously records the defect so that no downstream node
quotes it as validated resistance. **Fixing the transcription is out of scope
here** (it would change established anchors in `ice_anchor/` and the W2/W3
report numbers); it is logged as the next action in §7.

## 6. Verification status (honesty statement)

`primary_source_verified = false` for **all 16** assets. In the build
environment no paywalled standard (IACS UR I, IMO Polar Code, ISO 19906) and no
monograph (Keinonen 1996) could be opened: general web search was blocked and
the IACS site is JavaScript-rendered without retrievable PDFs. What *is* verified:

* every closed-form expression is re-evaluated here and dimension/limit-checked;
* the transcribed numbers reproduce the repository kernels bit-for-bit
  (`global_ice_pressure`, `ISO19906Pressure`, `ice_anchor.lindqvist_1989`,
  relative agreement `≤ 1.3e-16`);
* monotonicity/positivity of every pressure table is asserted.

What is *not* verified: that the standard clauses say exactly what the citation
claims. Each `verification_note` states this per asset, and `role` prevents a
`repo_transcription` from being used as truth. This is an honest `WITHHELD`
position on the standard text, not a pass.

## 7. Next actions

1. Re-verify `IACS_PC6_*` and `PC_I_A_*` against the primary instruments
   (UR I1/I2 2016, Polar Code MSC.385(94)) as soon as a copy is reachable, and
   flip `primary_source_verified` per asset when it is.
2. Resolve the Lindqvist breadth inconsistency (`R_c → C_c σ_c h² B`? or a
   breadth-normalised form) in `ice_anchor`, then promote a *new* asset id to
   `accuracy_reference` and retire the current regression anchor.
3. Add ice-tank series (AARC/HSVA/Arctech) as `measured_dataset` assets with the
   same schema once the digitised tables are available.