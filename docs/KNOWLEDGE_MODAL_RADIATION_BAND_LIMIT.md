# Knowledge: band-limit the acoustic spectrum of a truncated modal basis

**Rule (one line).** When a truncated modal basis is coupled to an acoustic
radiation model, the acoustic spectrum must be restricted to
`[f_min, ~2·f_max_retained]` **and** the band coverage of the modal-velocity
mean square must be verified (target > 90 %). Evaluating a frequency-dependent
radiation resistance on frequencies far above the retained modal band inflates
the radiated power by orders of magnitude.

## Context and applicability

* Applies to any chain of the form *load history → modal superposition →
  surface velocity → acoustic power/far field*, where the structural model is a
  truncated modal basis (hull girder, plate assembly, rotor set) and the
  radiation model is a Rayleigh/baffle or BEM radiator whose resistance
  `R(f)` grows with frequency.
* Applies to frequency-domain acoustic post-processing of a time-domain modal
  response (the usual engineering set-up). It does **not** replace a proper
  radiation-impedance (BEM/FEM-fluid) solution when broad-band high-frequency
  radiation is the question: there a fine structural model is required instead.
* Boundary: if the modal basis is *complete enough* that `f_valid` exceeds the
  frequency of interest, the rule is trivially satisfied.

## Evidence and provenance

* Node **P210W2-N06**, repository TensorPhysics, module
  `src/tensorphysics/ice_vib_noise/`, branch `agent/m11144`, commit
  "P210W2-N06: ice-induced hull vibration -> underwater radiated noise coupling
  chain" (2026-11-10).
* Measured in this task on the design case (130 m ice-class hull, 3 hull-girder
  modes at 0.71/1.95/3.83 Hz, 60 s ice crushing record at 200 Hz):

  | acoustic band used | radiated acoustic energy | ratio to `W_in` |
  |---|---|---|
  | 0.45–89 Hz (Nyquist-limited, no validity rule) | 224 214 J | 73 % |
  | 0.45–7.08 Hz (`2·f_3`, validity rule) | 2 200 J | 0.023 % |

  The first row is unphysical (cannot radiate 73 % of the mechanical input as
  sound); the cause is that above the retained modal band the modal
  coordinates no longer represent a physical surface-velocity *shape*, yet the
  mode-shape pattern is still multiplied by a resistance `R_i(f)` that has
  grown by 4-5 orders of magnitude (`R_1(0.7 Hz) = 26 N s/m` vs
  `R_1(6.7 Hz) ≈ 9e5 N s/m`).
* Artifacts: `artifacts/ice_vib_noise_coupling/report.json`
  (`criteria.C3_energy_conservation.acoustic_validity_band_hz`,
  `band_coverage_share_of_modal_velocity_ms`), `docs/ICE_VIB_NOISE_COUPLING_P210W2_N06.md` §5.3.

## Mechanism (why it fails)

1. A truncated basis has no local/higher-mode flexibility, so the projection of
   a localised load (bow contact) onto the low global modes is not the physical
   deformation at frequencies above the modal band.
2. The radiation resistance of a global mode shape *increases* with frequency
   (`σ ∝ (kL)^2` at low `kL`, then `σ → 1`), so the error is amplified exactly
   where the modal representation is least trustworthy.
3. The product `R_i(f)·|q̇_i(f)|²` therefore behaves like `f²·(small)` at high
   frequency and can dominate the band-integrated power.

## Recommended procedure

1. Set `f_valid = min(f_band_max, factor · f_highest_retained)` with
   `factor ≈ 2` (default in `CoupledCase.acoustic_validity_factor`), and refuse
   to report acoustic quantities above it.
2. Compute the radiation resistance on a log grid inside the band and
   interpolate in log-log (never linear-extrapolate).
3. Verify the **band coverage** of each modal velocity mean square
   (`Σ_bands <q̇_i²>_band / <q̇_i²>_time`); require > 90 %; report the number.
4. Report the residual of the energy balance with the *spectral* radiated
   energy and attribute it (Quadrature error vs radiation-model difference) —
   see `criteria.C3_energy_conservation.residual_attribution`.
5. State explicitly in the report that higher-frequency noise mechanisms
   (cavitation, local crushing, bubble release) are outside the model.

## Counterexamples / limitations

* Restricting the band can hide a genuine high-frequency problem; always pair
  the restriction with the coverage number and, if the coverage is low, refine
  the structural basis instead of widening the band.
* If the excitation frequency sits *above* `f_valid` (e.g. `f_ice = 6.7 Hz`
  with `f_3 = 3.83 Hz` in the first attempt of this node), the acoustic band
  contains no excitation line and the reported source spectrum is a structural
  resonance tail only. Move the *scenario* (flaking length/drift speed) rather
  than the band, or add modes.
* The 2·f_max rule is an engineering convention, not a theorem; it was adequate
  here (coverage 99.5 %/99.97 %/99.9 % for the three modes).

## Regression guard

* `tests/test_ice_vib_noise_chain.py::test_chain_physical_sanity` asserts
  `band_coverage_share_of_modal_velocity_ms > 0.9` for every mode **and**
  `E_rad < 1 % of W_in`. Both assertions fail loudly if the band limit is
  removed or widened without a structural justification.
* `criteria.C3_energy_conservation.acoustic_validity_rule` carries the rule text
  into every generated report, so the audit trail shows which band was used.

## Review status

Self-verified in-task (26 pytest cases, all criteria machine-checked); not yet
peer-reviewed by an independent seat. Revisit if a real BEM radiation model
replaces the baffle approximation.

**Last verified:** 2026-11-10 (node P210W2-N06 run, artifacts
`artifacts/ice_vib_noise_coupling/`).