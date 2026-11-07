# Migration guide: hand-rolled GCI -> `tensorphysics.vv`

This guide moves a solver repository's convergence verification from its own
ad-hoc Richardson/GCI snippet onto the shared machine gate, **without touching
the solver**: the repository keeps producing a JSON benchmark document, the gate
reads it from outside.

Target audience: whoever maintains a convergence/verification script in
TensorLBM, TensorFVM, TensorFEM, TensorDEM or TensorBEM.

Related: `docs/VV_GATE_API.md` (API), `docs/VV_GCI_ASME_GATE.md` (acceptance
evidence), `artifacts/vv_gate_n07/acceptance.json` (machine record).

---

## 0. Why migrate at all

Hand-rolled checks tend to test the wrong thing. Measured on the real
TensorLBM Poiseuille benchmark (`artifacts/vv_gate/obs_tensorlbm_poiseuille_*.json`):

| observable | repo's own criterion | repo verdict | gate verdict (theory `p = 2`) |
|---|---|---|---|
| relative L2 velocity error | mean refinement ratio `> 1.5`, i.e. `p > 0.58` | PASS | **FAIL** (`p = 0.751`) |
| flow-rate error | same | PASS | **FAIL** (`p = 0.885`) |
| centreline-velocity error | same | PASS | **FAIL** (`p = 0.811`) |

The refinement-ratio test cannot distinguish `O(h^0.8)` from `O(h^2)`; the gate
can, and it reports the same conclusion for three observables of the same fields
(so the finding is about the discrete solution, not about one metric).

A second reason: a single implementation of the formulas. For TensorFEM's
manufactured strain-energy study the gate's Richardson value and relative GCI
agree with that repository's independent `convergence_verification` module to
`0.0` (order differs by `4.4e-16`), which is only checkable because both sides
print the same quantities.

---

## 1. Six steps (any repository)

**Step 1 - pick the observable.** Prefer the engineering quantity of interest
(force, strain energy, added mass, frequency) over an error norm. An error norm
converges to zero, so its *relative* GCI is unbounded by construction; if you
must use one, pass `--gci-absolute`.

**Step 2 - produce at least three grids** with a documented refinement ratio
`r = h_coarse / h_fine >= 1.3` (ideally uniform, `r = 2`). Four or more grids
unlock the power-law residual and the order-consistency checks and certify the
noise floor.

**Step 3 - export the numbers as JSON** in the repository's own dialect. The
adapter understands the field names the repositories already use
(`mesh_size`/`dx`/`panel_size`/`resolution` for the grid,
`value`/`l2_error`/`relative_strain_energy_error`/`volume_relative_error` for the
observable), so the export is usually a two-line `json.dump` next to the existing
benchmark output:

```json
{
  "schema": "tensorlbm.convergence/1.0",
  "case": "dg_advection_mms_periodic_sin",
  "quantity": "l2_error",
  "levels": [
    {"resolution": 32,  "value": 2.314618692109e-03, "label": "nx=32"},
    {"resolution": 64,  "value": 5.706360949278e-04, "label": "nx=64"},
    {"resolution": 128, "value": 1.421538135247e-04, "label": "nx=128"}
  ]
}
```

**Step 4 - adjudicate it:**

```bash
python -m tensorphysics.vv.gci --benchmark bench/convergence.json \
       --repo tensorlbm --expected-order 2 --format table
```

**Step 5 - persist the verdict next to the benchmark output** so the repository's
verification docs cite a machine-readable record instead of a printed number:

```bash
python -m tensorphysics.vv.gci --benchmark bench/convergence.json --repo tensorlbm \
       --expected-order 2 --format json --out bench/convergence_gate.json
```

**Step 6 - decide how to treat WARN.** A three-grid study without per-level
`sigma` is capped at WARN by the conservative default, because three points and
three fitted parameters leave no residual to detect a noise floor. Either
declare a per-level `sigma` (round-off, sampling or measurement uncertainty - the
honest option), or add a fourth grid, or - only when the solver is deterministic
and the noise floor is known to be irrelevant - pass
`--allow-undeclared-noise`.

---

## 2. Per-repository recipes

### TensorLBM

* Source of truth today: `tensorlbm.spatial_convergence.assess_spatial_convergence`
  (monotonicity + observed order + extrapolated value) and the MMS test in
  `tests/test_dg_advection.py`.
* Export: a document with `resolution` (or `dx`) and `value`.
* Real example: `tools/vv_crossrepo_lbm_dgmms.py` reproduces the repository's P1
  DG-advection MMS on `nx = 32, 64, 128, 256` (one period, CFL 0.2, SSP-RK3) and
  writes `artifacts/vv_gate_n07/tensorlbm_dgmms/tensorlbm_dg_mms_native.json`.
  Result: `p = 2.0222` (`|p - 2| = 0.022 < 0.2`), absolute GCI band
  `4.35e-5`, verdict **PASS** - the first TensorLBM study to pass the gate.
* Note: the LBM Poiseuille case is *not* second order (section 0). Migrating it
  would fail the gate today; that failure is the useful output.

### TensorFVM

* Export: `h`/`dx` + `value`, or `nx`/`n_cells` + `l2_error`. The profile
  `--repo tensorfvm` accepts both.
* Recommended: one MMS (manufactured) study and one engineering QoI study, so
  the gate can show the order is recovered before the QoI is trusted.

### TensorFEM

* Source of truth today: `benchmarks/h_convergence_gci.py` ->
  `artifacts/h-convergence-gci/evidence.json` (`bending` and `manufactured`
  sub-studies, each with `mesh_size` + `relative_strain_energy_error` /
  `strain_energy`).
* Export: keep the evidence document as-is; the adapter selects the sub-study
  (`--study manufactured --value-key strain_energy`).
* Real example: `p = 1.96219`, relative GCI `0.0212927`, identical to the
  repository's own module (`artifacts/vv_gate_n07/tensorfem_hconv/evidence.json`).
* Keep the repository's own verification module; the gate is a second,
  independent implementation - agreement between the two is the point.

### TensorDEM

* Export: a `runs`/`levels` list with `dt` (or `resolution`) and the observable
  (contact force, penetration, packing fraction). The `--repo tensordem` profile
  maps `dt` straight onto `h`, so a timestep-refinement study needs no
  transformation.
* Order for a discrete-element scheme is not a textbook constant; set
  `--expected-order` only when the scheme's order is actually known, otherwise
  use the gate for asymptotics + GCI alone.

### TensorBEM

* Source of truth today: `docs/hydrostatic_verification.json`
  (`hemisphere_convergence`: 5 panel levels with `panel_size` and
  `volume_relative_error` / `gm_relative_error`).
* Export: keep the document; run
  `--repo tensorbem --study hemisphere_convergence --value-key volume_relative_error`.
* Real example: `p = 1.9387` on five panel levels
  (`artifacts/vv_gate_n07/acceptance.json`, criterion F5). Panel-method
  convergence is often not a clean power law at coarse panel counts; the gate
  then reports the residual/order-spread checks instead of silently averaging.

---

## 3. Translation table: old snippet -> gate call

| old code | replacement |
|---|---|
| `p = log((f1-f2)/(f2-f3))/log(r)` | `observed_order(levels)` -> `(p, source, residual)` |
| `f_ext = f3 + (f3-f2)/(r**p-1)` | `richardson_extrapolated(levels, p)` |
| `gci = Fs*abs((f3-f1)/f3)/(r**p-1)` | `gci(coarse, fine, p, safety_factor=Fs)` -> `(value, is_relative)` |
| "monotonic?" `if (f1-f2)*(f2-f3) > 0` | `study.convergence_type == "monotone"` |
| `assert rate > 1.5` (refinement ratio) | `evaluate(..., expected_order=2)` -> check `order-band/expected` |
| `U = sqrt(u1**2 + u2**2)` | `validation_uncertainty(u_num, u_input)` |
| `ok = abs(f_sim - f_exp) <= U` | `validate(f_sim, f_exp, u_num=..., u_input=...)` |

The gate replaces *the decision*, not your solver; keep the repository's own
diagnostics and treat a disagreement as a finding to investigate.

---

## 4. Pitfalls (learned the hard way)

1. **Relative GCI of an error norm is meaningless.** `|f_fine| -> 0` makes the
   band explode (the DG-MMS run reports ~123 % relative). Use `--gci-absolute`
   or switch to the QoI.
2. **A fixed fine grid hides the refinement ratio.** `Fs|Δf|/f/(r^p-1)` is
   invariant in `r` when the fine grid is fixed, so "GCI decreases with r" must
   be checked with the *coarse* grid anchored - that is what
   `sweep_refinement_ratio` does.
3. **Three grids cannot detect noise.** Three points, three fitted parameters,
   zero residual: a 1 % noise floor can look like a clean power law. Of 30
   seeded 1 % perturbations of a known-quadratic sequence, 12 used to pass
   silently; the conservative default now caps all 30 at WARN/FAIL
   (`acceptance.json`, F2).
4. **Two-grid studies** need `--assumed-order` and get `Fs = 3.0`; the order is
   not observable, and the gate says so in its notes.
5. **Non-uniform ratios are fine** (least squares), but the residual and
   order-spread checks need four or more levels; with three levels the fit is
   exact and no residual exists.
6. **The `|E| = U_val` boundary** is a PASS under ASME V&V 20; the gate compares
   it with a `1e-12` relative slack so that rounding of `sqrt(U_num^2+U_input^2)`
   does not flip it, and still reports the unslacked ratio.
7. **One observable is not a verdict.** Run the gate on every observable of the
   case; a study that passes on the QoI and fails on an error norm is telling you
   the norm is dominated by a different term.
8. **Do not import the gate inside a solver.** Export JSON, run the gate
   outside. That keeps the solver dependency-free and the migration reversible.

---

## 5. Migration checklist

- [ ] observable is the QoI (or `--gci-absolute` is used for an error norm),
- [ ] >= 3 grids, refinement ratio `r >= 1.3` documented in the benchmark doc,
- [ ] JSON document committed next to the benchmark output,
- [ ] `--repo`/`--study`/`--value-key` pinned explicitly (no reliance on auto
      detection in CI),
- [ ] per-level `sigma` declared, or the WARN is accepted and explained,
- [ ] `--out ..._gate.json` verdict attached to the repository's verification
      doc,
- [ ] the old hand-rolled assertion is either removed or explicitly documented as
      a weaker pre-check,
- [ ] the solver repository itself shows no diff (`git status --porcelain`).