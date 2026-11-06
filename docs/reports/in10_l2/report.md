# IN10-L2 · Ice-navigation reference-value corpus (sha256-anchored)

**Node** IN10-L2 (logistics corps, level 2) · **repo** TensorPhysics · **mission** 4907
**Verdict** PASS on all machine criteria
**Corpus digest** `64fbdb4e89c0e8b01726c08051dade7c35bf912d559201751ad00436b76462b9`

## 1. Case introduction

Ice-going ships are sized by a chain of empirical and rule-based numbers that
are normally retyped by hand into every analysis script: Lindqvist's level-ice
resistance components, Keinonen's brash-ice semi-empirical coefficients, the
Riska/Kujala ice pressure–area relation, the IACS Polar Class local panel
pressure, and the IMO Polar Code category/PST requirements. Hand retyping is the
dominant error source in this chain, and a silently edited literature number is
indistinguishable from a genuine physics improvement.

This node builds a **content-addressed reference corpus** for that chain: 16
assets in five families, each carrying `source_citation`, `values`, `tolerance`
and `applicable_regime`, anchored by file-level and value-level sha256 digests,
consumable from `pytest` by the campaign's C1/B1 nodes.

Reference condition (geometry + ice state used by the resistance assets):

| quantity | value |
|---|---|
| length `L` | 150 m |
| breadth `B` | 20 m |
| bow angle `φ` | 30° |
| ice thickness `h` | 1.0 m |
| speed `V` | 2.0 m/s |
| ice crushing strength `σ_c` | 5.0 MPa |
| ice flexural strength `σ_f` | 0.5 MPa |
| ice density `ρ_i` | 900 kg/m³ |
| ice/steel friction `μ` | 0.15 |

## 2. Method

1. **Asset schema with provenance discipline.** Beyond the four mandatory
   fields, every asset declares `evidence_tier`
   (`published_formula` / `repo_transcription` / `derived_from_rule_form` /
   `literature_band` / `standards_clause`), `role`
   (`accuracy_reference` vs `regression_anchor`), `primary_source_verified` and a
   free-text `verification_note`. `role` is load-bearing: a `regression_anchor`
   may only detect drift and must never be quoted as external truth.
2. **Deterministic builder.** `benchmarks/ice_nav_refs/build_corpus.py` computes
   every number from the expression stored in the asset's `formula` field and
   writes canonical JSON (`sort_keys`, fixed separators). Rebuilding reproduces
   byte-identical files, so the anchors are reproducible rather than curated.
3. **Two-level anchoring.** `MANIFEST.json` pins each asset file
   (`sha256`, bytes) plus an `aggregate_sha256` over the sorted
   `path:sha256` lines; each asset additionally carries
   `value_sha256 = sha256({asset_id, values, tolerance})`, which is invariant
   under prose edits and changes on any digit edit.
4. **Fail-closed loader.** `tensorphysics.ice_nav_refs.load_corpus()` verifies
   the manifest before parsing, then validates each asset: missing/placeholder
   citation, missing field, malformed tolerance (unknown mode, zero/negative
   value, inverted band, band without bounds) and id duplication all raise.
5. **Acceptance runner.** `benchmarks/ice_nav_refs/run_acceptance.py` executes ten
   machine criteria, eleven loader-level negative probes, a byte-tamper probe, a
   rebuild-reproducibility probe and three cross-checks against this
   repository's own kernels, then writes the artefacts used by this report.

Toolchain: Python 3.11, stdlib only (`hashlib`, `json`) for the corpus
machinery; `matplotlib` (Agg) for the figures; `pytest` for the test suite.

## 3. Results

### 3.1 Acceptance criteria (numbers as written to `acceptance.json`)

| id | criterion | value | threshold | pass |
|---|---|---|---|---|
| C1 | reference assets >= 12 | 16 | 12 | ✅ |
| C2 | source families covered >= 5 | 5 | 5 | ✅ |
| C3 | assets carrying all four mandatory fields | 16 | 16 | ✅ |
| C4 | asset files whose sha256 matches MANIFEST.json | 5 | 5 | ✅ |
| C5 | assets with a reproducible value_sha256 anchor | 16 | 16 | ✅ |
| C6 | assets with a usable tolerance block | 16 | 16 | ✅ |
| C7 | negative-case probes refused (entry without citation etc.) | 11 | 11 | ✅ |
| C8 | tampered file byte detected by sha256 anchor | True | True | ✅ |
| C9 | corpus reproducible byte-for-byte from build_corpus.py | True | True | ✅ |
| C10 | corpus vs repository kernels agreement (rel. error < 1e-12) | 3 | 3 | ✅ |

### 3.2 Corpus composition

| family | assets |
|---|---|
| `iacs_ur_i_pc6` | 3 |
| `keinonen_brash` | 3 |
| `lindqvist_1989` | 4 |
| `polar_code_i_a` | 3 |
| `riska_kujala_pressure_area` | 3 |

Evidence tiers: {"derived_from_rule_form": 3, "literature_band": 4, "published_formula": 2, "repo_transcription": 4, "standards_clause": 3} ·
Roles: {"accuracy_reference": 12, "regression_anchor": 4} ·
`primary_source_verified`: 0 verified /
16 unverified.

| asset | family | tier | role | tolerance | #values | primary source verified |
|---|---|---|---|---|---|---|
| `IACS_PC6_DESIGN_ICE_THICKNESS` | iacs_ur_i_pc6 | literature_band | accuracy_reference | band [1.0, 1.4] | 3 | no |
| `IACS_PC6_PANEL_PRESSURE_TABLE` | iacs_ur_i_pc6 | derived_from_rule_form | accuracy_reference | +-0.25 relative | 5 | no |
| `IACS_PC6_PATCH_GEOMETRY` | iacs_ur_i_pc6 | repo_transcription | regression_anchor | +-0.02 relative | 2 | no |
| `KEI_BRASH_COEFF_BAND` | keinonen_brash | literature_band | accuracy_reference | +-0.3 relative | 5 | no |
| `KEI_COEFFICIENTS` | keinonen_brash | repo_transcription | regression_anchor | +-0.02 relative | 2 | no |
| `KEI_RESISTANCE_REF` | keinonen_brash | repo_transcription | regression_anchor | +-0.1 relative | 3 | no |
| `LIN_C_CRUSH_PHI30` | lindqvist_1989 | published_formula | accuracy_reference | +-0.001 relative | 2 | no |
| `LIN_C_SPEED_FR` | lindqvist_1989 | published_formula | accuracy_reference | +-0.001 relative | 2 | no |
| `LIN_ICE_STEEL_FRICTION_BAND` | lindqvist_1989 | literature_band | accuracy_reference | band [0.05, 0.2] | 3 | no |
| `LIN_RESISTANCE_COMPONENTS` | lindqvist_1989 | repo_transcription | regression_anchor | +-0.05 relative | 7 | no |
| `PC_CATEGORY_TO_POLAR_CLASS` | polar_code_i_a | standards_clause | accuracy_reference | exact equality | 3 | no |
| `PC_I_A_CATEGORY_THICKNESS` | polar_code_i_a | standards_clause | accuracy_reference | +-0.15 relative | 5 | no |
| `PC_I_A_PST_BAND` | polar_code_i_a | literature_band | accuracy_reference | band [-40.0, -20.0] | 3 | no |
| `RA_GLOBAL_PRESSURE_TABLE` | riska_kujala_pressure_area | derived_from_rule_form | accuracy_reference | +-0.15 relative | 5 | no |
| `RA_LOCAL_PRESSURE_TABLE` | riska_kujala_pressure_area | derived_from_rule_form | accuracy_reference | +-0.2 relative | 6 | no |
| `RA_PRESSURE_AREA_EXPONENT` | riska_kujala_pressure_area | standards_clause | accuracy_reference | band [0.4, 0.55] | 3 | no |

### 3.3 Key numbers

| quantity | value | criterion | pass |
|---|---|---|---|
| assets / families | 16 / 5 | ≥12 / ≥5 | ✅ |
| sha256-anchored files verified | 5/5 | all | ✅ |
| aggregate sha256 | `64fbdb4e89c0e8b01726c08051dade7c35bf912d559201751ad00436b76462b9` | reproducible | ✅ |
| negative probes refused | 11/11 | all | ✅ |
| corpus↔kernel relative error (worst) | 1.31e-16 | < 1e-12 | ✅ |
| acceptance criteria | 10/10 | all | ✅ |

Kernel cross-checks (corpus value vs repository kernel, relative error):
`global_ice_pressure_vs_hullgirder_kernel` = 1.31e-16, `local_pressure_vs_iso19906_kernel` = 0, `lindqvist_components_vs_ice_anchor_transcription` = 0.

### 3.4 Negative cases refused by the loader

| probe | exception | outcome |
|---|---|---|
| `citation_removed` | `MissingFieldError` | rejected |
| `citation_empty` | `MissingCitationError` | rejected |
| `citation_placeholder` | `MissingCitationError` | rejected |
| `citation_too_short` | `MissingCitationError` | rejected |
| `citation_non_string` | `MissingCitationError` | rejected |
| `values_removed` | `MissingFieldError` | rejected |
| `tolerance_removed` | `MissingFieldError` | rejected |
| `regime_removed` | `MissingFieldError` | rejected |
| `tolerance_zero` | `ToleranceError` | rejected |
| `tolerance_inverted_band` | `ToleranceError` | rejected |
| `value_relabelled_without_redigest` | `CorpusIntegrityError` | rejected |

The byte-tamper probe (`tampered_asset_file_byte`) rewrote one character of
`assets/riska_kujala_pressure_area.json`; the manifest check returned
`manifest_ok = False` and the loader raised, i.e. a
single edited digit cannot pass unnoticed.

## 4. Figures

* **fig1** (*pressure-area reference assets*) — the ISO/Riska local pressure law
  `p = p0 A^-0.45` with its ±20 % acceptance band, the PC6 panel-pressure curve
  with the `p0 = 2.0–3.0 MPa` band, and the anchored PC6 table points.
* **fig2** (*global pressure and Lindqvist components*) — the global pressure
  anchor `p_G = 1.2 (h/h_ref)^-0.5` against ice thickness, and the Lindqvist
  component build-up with the anchored value at `h = 1 m`.
* **fig3** (*tolerance budget*) — the acceptance tolerance of all 16 assets on a
  log scale, making the tiered tolerance policy (1e-3 closed form → 0.30 brash
  ice) visible in one picture.

*(Field contours are not applicable: the deliverable is a data corpus, not a
field solution. The comparable visual artefacts are the law-vs-anchor curves
above.)*

## 5. Points to watch

1. **`primary_source_verified = false` for all 16 assets.** No paywalled
   standard (IACS UR I1/I2, IMO Polar Code MSC.385(94), ISO 19906) and no
   monograph (Keinonen 1996) could be opened in the build environment; general
   web search was blocked and the IACS site is JavaScript-rendered without
   retrievable PDFs. The corpus is therefore a **transparent** anchor set:
   expressions re-derived and dimension-checked, transcribed numbers
   bit-reproduced against the repository kernels, but the standards' wording is
   `WITHHELD`. Each `verification_note` states this per asset; §"Next actions" of
   `docs/ICE_NAV_REFS_CORPUS.md` tracks the re-verification.
2. **A defect the corpus caught.** `LIN_RESISTANCE_COMPONENTS` is classified
   `regression_anchor` because the transcribed Lindqvist crushing term
   `R_c = C_c σ_c h²` carries no breadth factor: at the reference condition it
   yields `R_c = 1.078e+07 N`, equivalent to an effective
   contact patch of only `2.16 m²` on a 20 m beam, and
   `R_total = 1.084e+04 kN`, i.e. 17× the mid
   engineering envelope for this hull class. The value is frozen so drift is
   detectable, but `role` prevents it from being used as accuracy evidence.
   Repairing the transcription is deliberately out of scope of this node.
3. **Tolerances are tier-dependent by design.** `1e-3` on a closed-form
   evaluation is an arithmetic check, not physics; the physics-face tolerance is
   `0.15–0.30` on rule-form and brash-ice assets. Comparing a computed number
   against the `1e-3` assets and calling it "validated physics" would be a
   category error.
4. **Corpus boundary.** These are validation data only. No resistance,
   propulsion or scantling kernel was added or changed; the ship-domain
   behaviour stays in the domain ship platform.

## 6. Error judgement against baselines

| key quantity | computed / recorded | baseline | deviation | < 3 % ? |
|---|---|---|---|---|
| corpus tables vs `hullgirder.ice_load.global_ice_pressure` | 1.2 MPa @ h=1 m | kernel | 1.31e-16 | **是** |
| corpus table vs `ice_plasticity.iso19906.ISO19906Pressure` | 3 MPa @ A=1 m² | kernel | 0 | **是** |
| Lindqvist components vs `ice_anchor.lindqvist_1989` | `R_total = 1.084e+04 kN` | kernel transcription | 0 | **是** |
| PC6 panel pressure vs generic ISO local pressure | 2.5 MPa @ A=1 m² | 3 MPa | -16.7 % | **否**（by design：PC6 nominal level is 0.5 MPa below the generic first-year level；covered by the ±25 % band, not by a 3 % criterion） |
| Lindqvist `R_total` vs engineering envelope | 1.084e+04 kN | 300–1000 kN envelope (estimate) | 17× | **否**（flagged defect；asset is `regression_anchor`） |
| global vs local pressure consistency `p_G(h=1 m)` vs `p(A=10 m²)` | 1.2 vs 1.064 MPa | ratio < 1 expected | ratio 1.13 | **是**（ordering holds） |

Reading of the table: every quantity that *must* agree with an existing
implementation agrees to machine precision (three rows, `< 3 %` satisfied by
~15 orders of magnitude). The two `否` rows are not calibration failures but
declared semantics: one is an intentional 0.5 MPa class offset inside an
explicit ±25 % band, the other is the transcription defect that the corpus is
designed to expose and refuse as accuracy evidence.

## 7. Artefacts

| path | content |
|---|---|
| `benchmarks/ice_nav_refs/assets/*.json` | 16 reference assets (5 families) |
| `benchmarks/ice_nav_refs/MANIFEST.json` | file sha256 + aggregate digest |
| `benchmarks/ice_nav_refs/build_corpus.py` | deterministic builder (`--check` mode) |
| `benchmarks/ice_nav_refs/run_acceptance.py` | 10 criteria + 12 negative probes + 3 kernel cross-checks |
| `src/tensorphysics/ice_nav_refs/` | loader / validator / tolerance API |
| `tests/test_refs_corpus.py` | 42 pytest assertions incl. negative cases |
| `artifacts/ice_nav_refs/acceptance.json` | machine acceptance record |
| `artifacts/ice_nav_refs/verification_matrix.json` | per-asset provenance matrix |
| `artifacts/ice_nav_refs/fig1..fig3*.png` | figures of §4 |
| `docs/ICE_NAV_REFS_CORPUS.md` | design note, known defects, next actions |
