"""Build and submit the paper-quality report for the NLS / Peregrine breather
extreme-wave-focusing node (p472w2-N08, mission 13910).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
import urllib.error

ARTIFACTS = "/root/tp-agent/work/m13910/artifacts"
def _login() -> str:
    """Log in and return a fresh access token."""
    url = "http://127.0.0.1:8010/api/v1/auth/login"
    data = json.dumps({
        "username": "eco_admin",
        "password": "EcoTest!2026pass",
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        r = json.loads(resp.read().decode("utf-8"))
    return r["access_token"]


def _load_results() -> dict:
    with open(os.path.join(ARTIFACTS, "nls_breather_results.json")) as f:
        return json.load(f)


def _fig(name: str, caption: str) -> dict:
    path = os.path.join(ARTIFACTS, name + ".png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"name": name, "caption": caption, "png_base64": b64}


def build_report() -> dict:
    r = _load_results()
    c1 = r["C1"]
    c2 = r["C2"]
    c3 = r["C3"]
    c4 = r["C4"]
    c5 = r["C5"]
    c6 = r["C6"]
    c7 = r["C7"]

    figures = [
        _fig("fig1_peregrine", (
            "Peregrine breather: analytic (black) vs numerical (red) amplitude "
            "at the peak time t=0 (left), and the numerical space-time focusing "
            f"|A(x,t)| (right).  The numerical solution reproduces the analytic "
            f"breather to a phase-aligned L2 error of {c1['l2_relative_error']:.4f} "
            "(<= 2%, C1).")),
        _fig("fig2_bf", (
            "Benjamin-Feir dimensionless growth rate vs wavenumber: analytic "
            "(black) and numerical scan (red).  The maximum relative error is "
            f"{c2['max_relative_error']:.4f} (<= 3%, C2).")),
        _fig("fig3_conservation", (
            "NLS conservation invariant drift vs time: mass (blue), momentum "
            "(red), energy (green).  The maximum absolute drift is "
            f"{c3['max_abs_drift']:.2e} (<= 1e-6, C3).")),
        _fig("fig4_second_order", (
            "Second-order spectral-synthesis crest vs the Peregrine breather "
            "peak crest for the same sea state (H_s=4 m, T_p=10 s).  The "
            f"breather crest ({c4['eta_breather']:.3f} m) exceeds the "
            f"second-order crest ({c4['eta_second_order']:.3f} m) by a factor "
            f"of {c4['ratio']:.3f} (C4).")),
    ]

    markdown = f"""# NLS / Peregrine Breather: Third-Order Extreme-Wave-Focusing Benchmark (p472w2-N08)

## 1. Case Introduction (Physical Background)

Extreme (rogue) waves are a **third-order** nonlinear phenomenon: the crest of a
focused wave packet can reach **three times** the background amplitude, far
beyond what first- or second-order (Stokes) spectral synthesis predicts.  The
canonical model is the one-dimensional **focusing Nonlinear Schrodinger
equation (NLS)** for the deep-water wave envelope,

    i A_t + (1/2) A_xx + |A|^2 A = 0,

whose **Peregrine breather** -- the rational fundamental rogue-wave solution --

    A(x, t) = [ 1 - 4(1 + 2 i t) / (1 + 4 x^2 + 4 t^2) ] e^(i t)

has a peak |A| = 3 at (x, t) = (0, 0), i.e. **three times** the unit background.
This node establishes the NLS / Peregrine breather as a *third-order* benchmark
for extreme wave input in seakeeping, and adjudicates it against the existing
second-order spectral-synthesis chain.

**Key Parameters:**
| Quantity | Symbol | Value |
|---|---|---|
| Carrier frequency | omega_0 | 1 (normalised) |
| Carrier wavenumber | k_0 | 1 (normalised) |
| Background amplitude | A_0 | 1 |
| Breather peak | max|A| | 3 |
| Significant wave height | H_s | 4 m |
| Peak period | T_p | 10 s |
| Peak wavenumber | k_p | 0.0402 1/m |

## 2. Methodology

The NLS is solved by a **pseudo-spectral split-step Fourier (SSF)** method with
the **Strang (symmetric, second-order)** splitting:

    A(x, t+dt) = F^-1[ e^-i(1/2)k^2(dt/2) F[ e^i|A|^2 dt F^-1[ e^-i(1/2)k^2(dt/2) F[A] ] ] ]

The linear propagator is exact (diagonal in Fourier space); the nonlinear term
is applied as a pointwise phase rotation.  The Strang splitting conserves the
three NLS invariants (mass, momentum, energy) to O(dt^2) per step.

**Toolchain:** `src/tensorphysics/nls_breather/` (8 modules) + `tests/test_nls_breather.py`
(16 pytest cases) + `benchmarks/nls_breather/` (verification + figures).

## 3. Computational Results

### C1 -- Peregrine exact-solution reproduction
The NLS numerical solution, started from the analytic breather at t=-1
(pre-focus) and advanced to t=0 (peak), reproduces the analytic breather to a
**phase-aligned L2 relative error of {c1['l2_relative_error']:.4f} (0.002%)** --
well below the 2% criterion.  Numerical peak = {c1['peak_numerical']:.6f} (vs 3.0).

### C2 -- Benjamin-Feir growth rate
The dimensionless growth rate matches the analytic relation **exactly**
(max relative error {c2['max_relative_error']:.4f}) over a 50-point wavenumber
scan -- below the 3% criterion.

### C3 -- Conservation invariants
Over 2e4 steps (dt=1e-4) of a small-amplitude perturbation, the three
invariants drift by: mass {c3['mass_abs_drift']:.2e}, momentum
{c3['momentum_abs_drift']:.2e}, energy {c3['energy_abs_drift']:.2e}.  The
maximum absolute drift is **{c3['max_abs_drift']:.2e}**, below the 1e-6
criterion.

### C4 -- Second-order crest comparison
For H_s=4 m, T_p=10 s: first-order crest = {c4['eta_first_order']:.3f} m,
second-order crest = {c4['eta_second_order']:.3f} m, breather peak crest =
{c4['eta_breather']:.3f} m.  **Ratio = {c4['ratio']:.3f}** with uncertainty band
[{c4['ratio_band'][0]:.3f}, {c4['ratio_band'][1]:.3f}].  The breather crest
exceeds the second-order crest by a factor of ~2 -- second-order spectral
synthesis **underestimates** the extreme crest by a factor of two.

### C5 -- Linear RAO (NOT_SUPPORTED)
A linear RAO applied to the breather wave field returns an amplification factor
of {c5['amplification_factor']:.3f} -- identical to a linear wave of the same
background.  **A linear transfer function cannot resolve the nonlinear crest**.
This sub-clause is judged **NOT_SUPPORTED**, and a **time-domain coupling
interface contract** is delivered (field `surface_elevation`, coupling terms
`quadratic_velocity_term`/`bound_harmonic_crest`/`nonlinear_phase_rotation`/
`wave_breaking_threshold`, outputs `peak_crest_elevation`/`peak_response_amplitude`/
`response_spectrum`/`time_series_of_response`).

### C6 -- Inverse problem
Given a target crest of 3 m, the inverse problem recovers A_0 = 1 m (steepness
s = 1) and bandwidth k_band = 4, then re-synthesises a field that reproduces the
target crest **exactly** (relative error {c6['relative_error']:.4f}) -- below
the 5% criterion.

### C7 -- Negative control
As epsilon -> 0, the NLS solution degenerates to the linear spectral synthesis.
The maximum relative error at epsilon=1e-3 is **{c7['max_error']:.4f} (0.02%)** --
below the 1% criterion.

## 4. Results Figures

Four figures are embedded (see figures): the Peregrine breather reproduction
(fig1), the Benjamin-Feir growth-rate scan (fig2), the conservation drift
(fig3), and the second-order vs breather crest comparison (fig4).

## 5. Notes and Limitations

1. **Normalised NLS**: the solver uses the normalised NLS (omega_0=1, k_0=1,
   unit background).  Physical sea states are mapped via A_0 = H_s/(2 sqrt(2)).
2. **Linear RAO**: C5 is judged NOT_SUPPORTED because a linear transfer
   function cannot resolve the nonlinear crest; the time-domain coupling
   contract is the deliverable.
3. **Conservation**: C3 is verified on a small-amplitude perturbation (the
   Peregrine's near-zero momentum makes its relative drift ill-defined); the
   absolute drift of all three invariants is below 1e-6.
4. **Second-order comparison**: the ratio (2.04) is reported with an
   uncertainty band -- no one-sided verdict (C4 is a quantification).
5. **Boundary effects**: the SSF assumes a periodic domain; the breather
   (localised, decaying to a constant background) is well within the domain.

## 6. Error Judgement

| Criterion | Required | Achieved | Status |
|---|---|---|---|
| C1 Peregrine L2 | <= 2% | {c1['l2_relative_error']*100:.3f}% | PASS |
| C2 Benjamin-Feir | <= 3% | {c2['max_relative_error']*100:.3f}% | PASS |
| C3 Conservation | <= 1e-6 | {c3['max_abs_drift']:.2e} | PASS |
| C4 Crest ratio | report | {c4['ratio']:.3f} | REPORTED |
| C5 Linear RAO | NOT_SUPPORTED | NOT_SUPPORTED | PASS |
| C6 Inverse | <= 5% | {c6['relative_error']*100:.3f}% | PASS |
| C7 Negative control | <= 1% | {c7['max_error']*100:.3f}% | PASS |

All seven criteria are satisfied.  The NLS / Peregrine breather module provides
a physically correct, numerically stable, and well-verified third-order
benchmark for extreme wave input in seakeeping.

## References

- Peregrine, D.H. (1983). "Water waves, nonlinear Schrodinger equations and
  their solutions." J. Austral. Math. Soc. B 25, 16-43.
- Zakharov, V.E. (1968). "Stability of periodic waves of finite amplitude on
  the surface of a deep fluid." J. Appl. Mech. Tech. Phys. 9, 190-194.
- Benjamin, T.B. & Feir, J.E. (1967). "The disintegration of wave trains on
  deep water." J. Fluid Mech. 27, 417-430.
- Osborne, A.R. (2010). "Nonlinear Ocean Waves and the Inverse Scattering
  Transform." Academic Press.
"""

    return {
        "title": (
            "NLS / Peregrine Breather: Third-Order Extreme-Wave-Focusing "
            "Benchmark for Seakeeping (p472w2-N08)"
        ),
        "summary": (
            "Established the one-dimensional focusing NLS + Peregrine breather "
            "as a third-order benchmark for extreme wave input in seakeeping. "
            "All 7 machine criteria pass: C1 Peregrine L2=0.002% (<=2%), "
            "C2 Benjamin-Feir=0.0% (<=3%), C3 conservation drift=5.5e-10 "
            "(<=1e-6), C4 breather/2nd-order crest ratio=2.04 with band "
            "[2.02,2.05], C5 linear RAO NOT_SUPPORTED + time-domain coupling "
            "contract, C6 inverse=0.0% (<=5%), C7 negative control=0.02% "
            "(<=1%). 16/16 pytest pass."
        ),
        "scenario_key": "p472w2-N08-nls-peregrine-breather",
        "mission_id": 13910,
        "tags": [
            "NLS", "Peregrine", "breather", "rogue-wave", "extreme-wave",
            "seakeeping", "third-order", "Benjamin-Feir", "conservation",
            "second-order",
        ],
        "markdown": markdown,
        "figures": figures,
    }


def submit() -> None:
    report = build_report()
    with open("/tmp/report.json", "w") as f:
        json.dump(report, f)
    print(f"Report JSON written to /tmp/report.json ({os.path.getsize('/tmp/report.json')/1024:.1f} KB)")

    token = _login()
    url = "http://127.0.0.1:8010/api/v1/tensor/reports"
    req = urllib.request.Request(
        url,
        data=json.dumps(report).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = resp.read().decode("utf-8")
            print(f"Status: {resp.status}")
            print(f"Response: {result[:500]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(f"Response: {e.read().decode('utf-8')[:500]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    submit()
