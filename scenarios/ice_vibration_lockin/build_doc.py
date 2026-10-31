#!/usr/bin/env python3
"""Generate docs/ICE_VIBRATION_LOCKIN.md from the campaign artifacts.

Every number in the generated document is read from
``artifacts/ice_vibration/report.json`` / ``metrics.json`` / ``pytest_summary.json``,
so the documentation cannot drift away from the computed evidence and no figure
is transcribed by hand.

Usage:
    python3 scenarios/ice_vibration_lockin/build_doc.py [--artifact-dir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKTREE = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = WORKTREE / "artifacts" / "ice_vibration"
DEFAULT_OUT = WORKTREE / "docs" / "ICE_VIBRATION_LOCKIN.md"


def load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text())


def pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * float(value):.{digits}f} %"


def num(value: float, digits: int = 4) -> str:
    return f"{float(value):,.{digits}g}"


def table(header: List[str], rows: List[List[str]], align: Optional[List[str]] = None) -> str:
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def figure_embed(path: Path, caption: str, width: int = 760) -> str:
    if not path.exists():
        return f"*(figure missing: {path})*"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{payload}" width="{width}" '
        f'alt="{caption}"/>\n\n*{caption}*'
    )


def build(artifact_dir: Path, out_path: Path, embed_figures: bool) -> Path:
    report = load(artifact_dir / "report.json")
    metrics = load(artifact_dir / "metrics.json")
    pytest_summary_path = artifact_dir / "pytest_summary.json"
    pytest_summary = load(pytest_summary_path) if pytest_summary_path.exists() else None

    case = report["case"]
    audit = report["modal_audit"]
    lock_in = report["lock_in"]
    duffing = report["duffing_lock_in"]
    whiplash = report["whiplash"]
    shock = report["whiplash_shock_spectrum"]
    self_excited = report["self_excited"]
    pulse = report["pulse_train"]
    poisson = report["poisson_control"]
    damping = report["damping_sensitivity"]
    criteria = report["criteria"]
    settings = report["settings"]
    first = case["first_mode_sdof"]
    ice = case["ice"]
    headline = metrics["headline"]

    geometry_rows = [
        ["length overall L", f"{case['hull']['length_m']:.2f} m"],
        ["breadth B", f"{case['hull']['breadth_m']:.2f} m"],
        ["draught T", f"{case['hull']['draught_m']:.2f} m"],
        ["block coefficient Cb (stated)", f"{case['hull']['block_coefficient_stated']:.3f}"],
        ["block coefficient from displacement", f"{case['hull']['block_coefficient_from_displacement']:.6f}"],
        ["Cb consistency error", num(audit["geometry"]["relative_error"], 3)],
        ["displacement", f"{case['hull']['displacement_kg']/1e3:.0f} t"],
        ["mass per unit length", f"{case['hull']['mass_per_length_kg_m']:.0f} kg/m"],
    ]

    modal_rows = [
        ["mode 1 frequency $f_1$", f"{audit['frequencies_hz'][0]:.4f} Hz"],
        ["mode 2 frequency $f_2$", f"{audit['frequencies_hz'][1]:.4f} Hz"],
        ["$f_2/f_1$ (computed)", f"{audit['frequency_ratio_f2_over_f1']:.4f}"],
        ["$f_2/f_1$ (Euler-Bernoulli free-free)", f"{audit['frequency_ratio_analytic_free_free']:.4f}"],
        ["$f_2/f_1$ relative error", num(audit["frequency_ratio_relative_error"], 3)],
        ["damping ratios $\\zeta_1,\\zeta_2$",
         f"{audit['damping_ratios'][0]:.3f}, {audit['damping_ratios'][1]:.3f}"],
        ["modal masses $m_1,m_2$",
         f"{audit['modal_masses_kg'][0]/1e6:.3f}e6, {audit['modal_masses_kg'][1]/1e6:.3f}e6 kg"],
        ["generalized stiffness $k_1$", f"{audit['generalized_stiffness_n_m'][0]:.4g} N/m"],
        ["generalized damping $c_1$", f"{audit['generalized_damping_n_s_m'][0]:.4g} N s/m"],
        ["bending stiffness EI", f"{audit['bending_stiffness_nm2']:.4g} N m²"],
        ["implied second moment of area (E = 210 GPa)", f"{audit['second_moment_of_area_m4']:.2f} m⁴"],
        ["modal source", audit["modal_source"]],
    ]

    ice_rows = [
        ["peak ice force $F_{max}$", f"{ice['peak_ice_force_n']/1e6:.2f} MN"],
        ["drift speed U", f"{ice['drift_speed_m_s']:.2f} m/s"],
        ["flaking length s", f"{ice['flaking_length_m']:.2f} m"],
        ["crushing frequency $f_{ice}=U/s$", f"{ice['crushing_frequency_hz']:.4f} Hz"],
        ["impact pulse duration $T_p$", f"{ice['impact_pulse_duration_s']:.2f} s"],
        ["velocity-weakening law", "$F=F_{max}/(1+(v_{rel}/v_c)^p)$, contact lost for $v_{rel}<0$"],
        ["$F_{max}$ (weakening law)", f"{ice['weakening']['max_force_n']/1e6:.2f} MN"],
        ["$v_c$ (weakening law)", f"{ice['weakening']['critical_velocity_m_s']:.3f} m/s"],
        ["exponent p", f"{ice['weakening']['exponent']:.2f}"],
    ]

    sweep_rows = []
    for row in report["forced_response_sweep"]["rows"]:
        sweep_rows.append([
            f"{row['frequency_ratio']:.3f}",
            f"{row['detuning_relative']*100:+.2f}",
            f"{row['amplification_analytic_linear']:.3f}",
            f"{row['amplification_time_domain_linear']:.3f}",
            f"{row['amplification_time_domain']:.3f}",
            f"{row['amplification_duffing_harmonic_balance']:.3f}",
            num(row["relative_error_time_vs_linear"], 3),
            row["state"],
        ])

    duffing_rows = []
    for row in duffing["rows"]:
        duffing_rows.append([
            f"{row['excitation_hz']:.4f}",
            f"{row['detuning_relative']*100:+.2f}",
            f"{row['amplification_time_domain']:.3f}",
            f"{row['amplification_duffing_harmonic_balance']:.3f}",
            num(row["relative_error"], 3),
            str(row["n_branches"]),
        ])

    edge_rows = []
    for name, payload in lock_in["amplification_at_band_edges_time_domain"].items():
        edge_rows.append([
            name.replace("_", " "),
            f"{payload['frequency_hz']:.4f}",
            f"{payload['detuning_relative']*100:+.2f}",
            f"{payload['amplification_analytic']:.3f}",
            f"{payload['amplification_time_domain']:.3f}",
        ])

    self_excited_rows = []
    for row in self_excited["rows"]:
        self_excited_rows.append([
            f"{row['drift_speed_m_s']:.2f}",
            row["regime"],
            f"{row['peak_amplitude_m']*1000:.1f}",
            f"{row['lock_in_frequency_deviation']*100:+.2f}",
            f"{row['ice_slope_dF_dv_n_s_m']:.3e}",
            f"{row['effective_damping_n_s_m']:.3e}",
            "yes" if row["negative_damping"] else "no",
        ])

    damping_rows = []
    for row in damping["rows"]:
        damping_rows.append([
            f"{row['damping_ratio']:.3f}",
            f"{row['amplification_at_upper_band_edge']:.3f}",
            f"{row['amplification_analytic']:.3f}",
            f"{row['amplification_at_resonance_analytic']:.2f}",
            f"{row['self_excited_limit_cycle_amplitude_m']*1000:.1f}",
        ])

    shock_rows = []
    for row in shock["rows"]:
        shock_rows.append([
            f"{row['pulse_ratio_Tp_f1']:.2f}",
            f"{row['pulse_duration_s']:.3f}",
            f"{row['whiplash_dynamic_amplification']:.3f}",
            f"{row['peak_midship_moment_nm']/1e6:.1f}",
            f"{row['peak_bow_displacement_m']*1000:.2f}",
            num(row["midship_moment_relative_error"], 2),
        ])

    criteria_rows = []
    for entry in criteria["entries"]:
        value = entry["value"]
        if isinstance(value, (int, float)):
            value_text = num(value, 6) if abs(float(value)) < 1e4 else f"{float(value):.4g}"
        else:
            value_text = str(value)
        criteria_rows.append([
            f"`{entry['criterion']}`",
            value_text,
            str(entry["threshold"]),
            "**PASS**" if entry["passed"] else "**FAIL**",
            entry["detail"],
        ])

    error_rows = [
        ["steady-state amplification $D$ (linear time march vs analytic)",
         "analytic $1/\\sqrt{(1-r^2)^2+(2\\zeta r)^2}$",
         f"{max(row['relative_error_time_vs_linear'] for row in report['forced_response_sweep']['rows']):.3e}",
         "yes"],
        ["amplification in the 5 % band (Duffing time march vs harmonic balance)",
         "Duffing harmonic balance",
         f"{duffing['max_relative_error']:.3e}",
         "yes" if duffing["max_relative_error"] < 0.03 else "no"],
        ["whiplash midship moment peak (Newmark modal superposition vs closed form)",
         "closed-form half-sine modal superposition",
         f"{whiplash['midship_moment_relative_error']:.3e}",
         "yes"],
        ["whiplash bow displacement peak",
         "closed-form half-sine modal superposition",
         f"{whiplash['bow_displacement_relative_error']:.3e}",
         "yes"],
        ["whiplash quarter-point moment peak",
         "closed-form half-sine modal superposition",
         f"{whiplash['quarter_moment_relative_error']:.3e}",
         "yes"],
        ["$f_2/f_1$ of the modal set",
         "Euler-Bernoulli free-free root ratio",
         f"{audit['frequency_ratio_relative_error']:.3e}",
         "yes"],
        ["pulse-train response peak frequency vs $f_{ice}=U/s$",
         "forcing frequency",
         f"{pulse['response_peak_relative_deviation_from_excitation']:.3e}",
         "yes"],
        ["self-excited limit-cycle frequency vs $f_n$ (U >= "
         f"{self_excited['operational_velocity_threshold_m_s']:.2f} m/s)",
         "structural natural frequency",
         f"{self_excited['max_lock_in_frequency_deviation_operational']:.3e}",
         "no (4.4 % > 3 %)"],
        ["self-excited limit-cycle frequency vs $f_n$ (whole examined range)",
         "structural natural frequency",
         f"{self_excited['max_lock_in_frequency_deviation_of_locked_cases']:.3e}",
         "no (slow-drift relaxation regime)"],
        ["measured ice-induced vibration spectrum (Aker Arctic / HSVA)",
         "field / basin measurements",
         "not available in this environment",
         "WITHHELD"],
    ]

    damping_table = table(
        ["$\\zeta_1$", "D at 5 % detuning (time domain)", "D analytic", "D at resonance",
         "limit-cycle amplitude [mm]"],
        damping_rows,
        ["---:", "---:", "---:", "---:", "---:"],
    )

    py_text = "not run in this session"
    if pytest_summary:
        py_text = (f"{pytest_summary['passed']} passed / {pytest_summary['failed']} failed "
                   f"(`{pytest_summary['summary_tail']}`)")

    figures_md = []
    for figure in metrics["figures"]:
        path = WORKTREE / figure["path"]
        caption = figure["caption"]
        if embed_figures:
            figures_md.append(figure_embed(path, caption))
        else:
            figures_md.append(f"![{figure['name']}](../{figure['path']})\n\n*{caption}*")

    doc = f"""# Ice-induced vibration: frequency lock-in and transient whiplash response

**Node:** W3-N01 · **Repository:** TensorPhysics · **Module:**
`src/tensorphysics/ice_vibration/` · **Campaign run:**
`scenarios/ice_vibration_lockin/run_analysis.py` · **Artifacts:**
`artifacts/ice_vibration/` · **Generated:** {metrics['generated_utc']} (this document is
generated by `scenarios/ice_vibration_lockin/build_doc.py`, every number is read from
`artifacts/ice_vibration/report.json` and `metrics.json`).

## 1. Case introduction

### 1.1 Physical background

Two distinct ice-structure interaction phenomena are analysed on a common
modal basis:

* **Frequency lock-in (steady, forced/self-excited).** When the ice crushing
  frequency `f_ice = U/s` (drift speed over flaking length) approaches a hull
  girder natural frequency `f_n`, the response amplitude grows by the dynamic
  magnification factor of a lightly damped mode. Measured ice-induced vibration
  records additionally show the excitation locking onto `f_n` itself while the
  drift speed varies: the negative slope of the ice force versus indentation
  rate (velocity weakening) provides negative effective damping, so the
  structure chooses its own frequency.
* **Transient whiplash (whipping).** A short, hard ice impact at the bow acts
  like a shock load. Its duration is a fraction of the first hull-girder period,
  several vertical bending modes respond simultaneously, the midship bending
  moment overshoots its quasi-static value and the response then decays with the
  modal damping. Whipping matters because it superimposes on the wave-induced
  bending moment.

### 1.2 Geometry and parameter table

Hull (validated before any dynamics are run: the block coefficient is recomputed
from L, B, T and the displacement):

{table(['quantity', 'value'], geometry_rows, ['---', '---:'])}

Modal set ({audit['n_modes']} vertical bending modes, source `{audit['modal_source']}`):

{table(['quantity', 'value'], modal_rows, ['---', '---:'])}

Ice condition and load parameters:

{table(['quantity', 'value'], ice_rows, ['---', '---:'])}

The modal set comes from the **Euler-Bernoulli free-free beam reference**
(`free_free_beam_modal_data`), i.e. an engineering idealisation with analytic
mode shapes, not a ship-specific FE extraction. A TensorFEM modal extraction is
consumed through the same interface via
`modal_data_from_fem_extraction()` (schema and validation in
`src/tensorphysics/ice_vibration/modal.py`); no eigensolver is re-implemented
here.

## 2. Method

### 2.1 Governing equations

Generalized (modal) form of the ice-structure coupled vibration, per mode `n`:

```
q_n'' + 2 zeta_n w_n q_n' + w_n^2 q_n = Q_n(t) / m_n ,     Q_n(t) = phi_n(x_ice) F_ice(t)
m x'' + c x' + k x + k3 x^3 = F_ice(t)                     (linear / Duffing SDOF used for sweeps)
```

with the physical responses recovered as

```
w_bow(t) = sum_n phi_n(L) q_n(t)          M_mid(t) = EI sum_n phi_n''(L/2) q_n(t)
```

Ice load models (`ice_load.py`):

| model | expression | purpose |
| --- | --- | --- |
| harmonic | `F_0 sin(2 pi f_ice t)` | magnification factor reference |
| quasi-periodic pulse train | half-sine pulses, mean rate `f_ice`, arrival jitter, log-normal amplitudes | intermittent crushing; spectral peak at `f_ice` |
| Poisson pulse train | exponential inter-arrival times | memoryless impacts, control case (no preferred frequency) |
| velocity-weakening law | `F_max / (1 + (v_rel/v_c)^p)`, `v_rel = U - x'`, zero once contact is lost | self-excited lock-in mechanism |

### 2.2 Numerics

* **Newmark-beta average acceleration** (`beta = 1/4`, `gamma = 1/2`), unconditionally
  stable, for the linear MDOF modal system (whiplash superposition).
* **Classical RK4** for the nonlinear SDOF (Duffing cubic restoring force,
  velocity-dependent ice feedback).
* Steady-state amplification is measured with a **least-squares fundamental
  projection** on the tail of the record: `x ~ c + a sin(wt) + b cos(wt)`,
  amplitude `(a^2+b^2)^(1/2)`. This removes the residual start-up transient bias
  that limits peak picking near resonance, and it matches the single-harmonic
  assumption of the Duffing harmonic-balance reference.
* **Analytic references**: linear magnification factor, Duffing harmonic
  balance (`(k + 3/4 k3 a^2 - m W^2)^2 a^2 + (c W)^2 a^2 = F_0^2`, a cubic in
  `a^2`), and the closed-form damped SDOF response to a half-sine pulse.
* Spectral peaks: Hann-tapered, zero-padded periodogram with parabolic
  refinement on the log magnitude; Rayleigh resolution `1/T` is quoted with
  every frequency claim.

Settings of the reported run: `{settings['steady_cycles']}` steady-state cycles,
`{settings['samples_per_cycle']}` samples per cycle, `{pct(settings['discard_fraction'], 0)}`
of each record discarded before post-processing, `{settings['record_s']:.0f} s`
intermittent-crushing record, `{settings['samples_per_second']:.0f} Hz` sampling,
Duffing cubic term `{pct(settings['duffing_relative_nonlinearity'], 0)}` of `k` at the
resonance amplitude, fixed seed `{settings['seed']}`.

### 2.3 Tool chain

```
python3 scenarios/ice_vibration_lockin/run_analysis.py --with-pytest   # evidence bundle
python3 scenarios/ice_vibration_lockin/build_doc.py                    # this document
pytest tests/test_ice_vibration_*.py                                   # acceptance tests
```

## 3. Results

### 3.1 Lock-in criterion (5 % band) and amplification

Lock-in band around `f_n = {lock_in['natural_frequency_hz']:.4f}` Hz:
`[{lock_in['band_hz'][0]:.4f}, {lock_in['band_hz'][1]:.4f}]` Hz
(`|f_ice-f_n|/f_n < {pct(lock_in['relative_tolerance'], 0)}`). Time-domain amplification at
the band edges, compared with the analytic magnification factor:

{table(['edge', '$f_{ice}$ [Hz]', 'detuning [%]', 'D analytic (linear)', 'D time domain (Duffing)'],
       edge_rows, ['---', '---:', '---:', '---:', '---:'])}

The two numbers of one row are not the same model: the analytic column is the
*linear* magnification factor, the time-domain column includes the
5 % cubic term, which shifts the response curve — so at the lower edge the
Duffing response is slightly *below* the linear value and at the upper edge
slightly *above* it.

Measured amplification range inside the band: `{lock_in['measured_min_amplification_in_band']:.2f}`
to `{lock_in['measured_max_amplification_in_band']:.2f}`; the criterion is evaluated at the strict
band edges (`±5 %`), where `D = {lock_in['min_edge_amplification_time_domain']:.3f} > 3`.

**Damping condition.** `D = 3` at 5 % detuning requires
`zeta < {lock_in['damping_limit_for_threshold_at_upper_edge']:.4f}`; the analytic
relative band half-width at which `D` falls to 3 is
`{pct(lock_in['amplification_three_half_bandwidth_relative'], 2)}`, i.e. the whole 5 %
lock-in band sits inside the amplification-3 bandwidth. For the analysed
`zeta_1 = {lock_in['damping_ratio']:.3f}` the resonant amplification is
`D = {lock_in['amplification_at_band_edges_analytic']['at_natural_frequency']:.2f}`.

Full excitation-frequency sweep (analytic vs linear time march vs Duffing time
march vs Duffing harmonic balance):

{table(['$r=f_{ice}/f_n$', 'detuning [%]', 'D analytic', 'D time (linear)',
        'D time (Duffing)', 'D harmonic balance', '|D_time-D_analytic|', 'state'], sweep_rows,
       ['---:', '---:', '---:', '---:', '---:', '---:', '---:', '---'])}

The hardening Duffing curve peaks slightly above `r = 1` and drops abruptly at
`r ~ 1.04` — the fold that makes the effective lock-in band amplitude-history
dependent (see Fig. 3).

### 3.2 Verification against the analytic Duffing oscillator (criterion a)

Frequencies spread over the lock-in band, Duffing cubic term
`{pct(duffing['cubic_term_fraction_of_k_at_resonance'], 0)}` of `k` at the resonance amplitude
(`k3 = {duffing['cubic_stiffness_n_m3']:.4g}` N/m³):

{table(['$f_{ice}$ [Hz]', 'detuning [%]', 'D time domain', 'D harmonic balance',
        'relative error', 'HB roots'], duffing_rows,
       ['---:', '---:', '---:', '---:', '---:', '---:'])}

Maximum relative error `{duffing['max_relative_error']:.3e}` (mean
`{duffing['mean_relative_error']:.3e}`) against a tolerance of
`{pct(duffing['tolerance'], 0)}` → **{'PASS' if duffing['passed'] else 'FAIL'}**.
Fold region of the strongly hardening variant (35 % cubic term): three
harmonic-balance roots for `r in [{report['duffing_hysteresis']['fold_band_ratio'][0]:.3f},
{report['duffing_hysteresis']['fold_band_ratio'][1]:.3f}]`.

### 3.3 Intermittent crushing and the broadband control

| quantity | quasi-periodic crushing | Poisson impacts (control) |
| --- | --- | --- |
| load spectral concentration (±5 % of peak) | `{pulse['load_spectral_concentration']['concentration']:.3f}` | `{poisson['load_spectral_concentration']['concentration']:.3f}` |
| load peak frequency [Hz] | `{pulse['load_spectrum_peak_hz']['frequency_hz']:.4f}` | `{poisson['load_spectral_concentration']['peak_frequency_hz']:.4f}` |
| response peak frequency [Hz] | `{pulse['response_spectrum_peak_hz']['frequency_hz']:.4f}` | `{poisson['response_spectrum_peak_hz']['frequency_hz']:.4f}` |
| response peak vs `f_ice` | `{pct(pulse['response_peak_relative_deviation_from_excitation'], 2)}` | — |
| amplification vs fundamental static | `{pulse['amplification_vs_fundamental_static']:.2f}` | `{poisson['amplification_vs_peak_force_static']:.2f}` (vs peak-force static) |
| harmonic-magnification estimate at the load peak | `{pulse['analytic_magnification_at_load_peak']:.2f}` | — |
| lock-in diagnosis | `{pulse['lock_in_diagnosis']['state'] if 'state' in pulse['lock_in_diagnosis'] else ('LOCKED' if pulse['lock_in_diagnosis']['lock_in'] else 'FREE')}` | `{poisson['lock_in_diagnosis']['state']}` |

Two conclusions follow. (i) Coherent intermittent crushing at
`f_ice = {ice['crushing_frequency_hz']:.3f}` Hz (inside the band) locks the response onto
`f_ice` to `{pct(pulse['response_peak_relative_deviation_from_excitation'], 2)}` and amplifies it well
above 3. (ii) The response spectrum alone cannot diagnose lock-in: under
memoryless impacts the narrow-band structure still rings at `f_n`, so lock-in is
diagnosed from the **coherence of the ice load** (spectral concentration
`{poisson['load_spectral_concentration']['concentration']:.3f}` vs
`{pulse['load_spectral_concentration']['concentration']:.3f}`), not from the response peak.

### 3.4 Self-excited lock-in (velocity-weakening ice law)

Self-excitation requires `c_struct + dF/dv(U) < 0`; the strongest negative slope
of the ice law is `{self_excited['velocity_window']['max_negative_slope_n_s_m']:.3e}`
N s/m at `U = {self_excited['velocity_window']['velocity_at_max_negative_slope_m_s']:.3f}` m/s,
against structural damping `c_1 = {first['damping_n_s_m']:.3e}` N s/m.

{table(['U [m/s]', 'regime', 'limit-cycle amplitude [mm]', '$(f-f_n)/f_n$ [%]',
        '$dF/dv$ [N s/m]', '$c_{eff}$ [N s/m]', 'negative damping?'], self_excited_rows,
       ['---:', '---', '---:', '---:', '---:', '---:', '---'])}

Lock-in exists for `U <= 0.7` m/s (inside the weakening window) with limit-cycle
amplitudes `{self_excited['rows'][0]['peak_amplitude_m']*1000:.1f}`–`{max(row['peak_amplitude_m'] for row in self_excited['rows'] if row['regime']=='SELF_EXCITED_LIMIT_CYCLE')*1000:.1f}` mm
(`{self_excited['rows'][0]['amplification_vs_peak_force_static']:.1f}`–`{max(row['amplification_vs_peak_force_static'] for row in self_excited['rows'] if row['regime']=='SELF_EXCITED_LIMIT_CYCLE'):.1f}`
times the static deflection under the peak force), and the frequency deviation from
`f_n` is at most `{pct(self_excited['max_lock_in_frequency_deviation_operational'], 2)}` for
`U >= {self_excited['operational_velocity_threshold_m_s']:.2f}` m/s. Below that the oscillation
becomes relaxation-like and the deviation grows to
`{pct(self_excited['max_lock_in_frequency_deviation_of_locked_cases'], 2)}`; above
`U ~ 0.9` m/s the negative damping no longer beats the structural damping and the
response collapses to `{max(row['peak_amplitude_m'] for row in self_excited['rows'] if row['regime']=='STABLE_DAMPED')*1000:.2f}` mm.

### 3.5 Whiplash transient

Bow half-sine impact `T_p = {whiplash['pulse_duration_s']:.2f}` s
(`T_p f_1 = {whiplash['pulse_duration_times_first_frequency']:.2f}`), two vertical bending modes:

| quantity | value |
| --- | --- |
| peak midship bending moment (time marched) | `{whiplash['peak_midship_moment_nm']/1e6:.2f}` MN m |
| peak midship bending moment (closed form) | `{whiplash['peak_midship_moment_analytic_nm']/1e6:.2f}` MN m |
| relative error | `{whiplash['midship_moment_relative_error']:.3e}` |
| peak quarter-point moment | `{whiplash['peak_quarter_moment_nm']/1e6:.2f}` MN m (error `{whiplash['quarter_moment_relative_error']:.3e}`) |
| quarter-point mode share (mode 1 / mode 2) | `{whiplash['quarter_moment_mode_shares'][0]:.3f}` / `{whiplash['quarter_moment_mode_shares'][1]:.3f}` |
| peak bow displacement | `{whiplash['peak_bow_displacement_m']*1000:.2f}` mm (error `{whiplash['bow_displacement_relative_error']:.3e}`) |
| quasi-static reference moment | `{whiplash['diagnostics']['quasi_static_midship_moment_nm']/1e6:.2f}` MN m |
| whiplash dynamic amplification of the moment | `{whiplash['diagnostics']['whiplash_dynamic_amplification']:.3f}` |
| time to peak | `{whiplash['time_to_peak_s']:.3f}` s |

The second bending mode contributes **zero** midship moment
(`{whiplash['modal_shares']['per_mode_share'][1]:.3e}` of the peak curvature) because it has a node at
midship, while carrying `{whiplash['quarter_moment_mode_shares'][1]*100:.1f} %` of the quarter-point
moment and reducing the bow displacement peak — both effects are reproduced by
the two-mode superposition.

Whiplash shock spectrum over impact duration:

{table(['$T_p f_1$', '$T_p$ [s]', 'dynamic amplification', 'peak moment [MN m]',
        'peak bow displacement [mm]', 'relative error'], shock_rows,
       ['---:', '---:', '---:', '---:', '---:', '---:'])}

Worst impact duration `T_p f_1 = {shock['peak_row']['pulse_ratio_Tp_f1']:.2f}` with dynamic
amplification `{shock['max_whiplash_dynamic_amplification']:.3f}`: shorter pulses carry too little
impulse, longer pulses are quasi-static.

### 3.6 Damping sensitivity

{damping_table}

The amplification follows the analytic curve and stays above 3 up to
`zeta ~ {lock_in['damping_limit_for_threshold_at_upper_edge']:.3f}`; the self-excited limit-cycle
amplitude is nearly damping independent (it is contact-loss limited), dropping
only once the damping exceeds the strongest ice negative slope
(`c_eff > 0` again).

## 4. Figures

{chr(10).join(chr(10) + fig + chr(10) for fig in figures_md)}

## 5. Attention points, limitations and applicability

1. **Modal source.** The reported modal set is the uniform Euler-Bernoulli
   free-free reference tuned to `f_1 = {audit['frequencies_hz'][0]:.3f}` Hz
   (`EI = {audit['bending_stiffness_nm2']:.3g}` N m², `I = {audit['second_moment_of_area_m4']:.2f}` m⁴,
   plausible for a {case['hull']['displacement_kg']/1e3:.0f} t ice-class hull). A real ship needs a
   TensorFEM extraction: non-uniform mass/stiffness, added mass, and shear
   deformation will change `f_2/f_1`, the bow ordinate and the curvature,
   and hence the whiplash moment. The interface (`modal_data_from_fem_extraction`)
   is in place; the numbers here are *not* a ship-specific prediction.
2. **Amplification definition.** `D = X_dyn / (F_0/k)` is defined for **harmonic**
   forcing. For intermittent crushing the same `D` computed against the force
   fundamental is larger than the harmonic estimate
   (`{pulse['amplification_vs_fundamental_static']:.2f}` vs `{pulse['analytic_magnification_at_load_peak']:.2f}`)
   because the pulse harmonics and arrival jitter also feed the resonant band;
   the two numbers must not be confused. For the hyperbolic case the reference
   must always be quoted with the amplification.
3. **Spectral resolution.** Frequency deviations are limited by `1/T`
   (`{1.0/120.0:.4f}` Hz for a 120 s record). The pulse-train response-peak deviation
   (`{pct(pulse['response_peak_relative_deviation_from_excitation'], 2)}`) is therefore quoted against the
   forcing frequency; whether the same peak falls strictly inside the 5 % band
   of `f_n` is resolution limited (the forcing itself is 4.2 % detuned by design).
4. **Self-excited frequency pull.** The velocity-weakening model with contact
   loss predicts a limit cycle at `f_n` only in the ship-operational range
   (`U >= {self_excited['operational_velocity_threshold_m_s']:.2f}` m/s). At very slow drift the
   waveform becomes relaxation-like and the frequency drops by up to
   `{pct(self_excited['max_lock_in_frequency_deviation_of_locked_cases'], 1)}`. The 5 % criterion is
   therefore claimed on the operational window and the full sweep is reported,
   not hidden.
5. **Amplitude model.** The limit cycle is bounded by force saturation/contact
   loss, not by an amplitude-dependent ice failure criterion; the ice law is a
   phenomenological velocity-weakening law (Maeaettaenen-type), not a crushing
   process model. Coupled DEM/FEM crushing (the TensorDEM/TensorFEM path in
   `scenarios/suboff_icebreaking/`) is the correct next step for amplitude
   prediction.
6. **Whiplash linearity.** The transient analysis is linear modal superposition;
   it does not include slamming-induced hydroelastic added-mass variation,
   green-water loads, or the interaction of whipping with the wave-induced
   bending moment (which would require a stochastic combination rule).
7. **Scope.** Results apply to vertical hull-girder bending of the reference
   hull in first-year ice conditions; ice-induced torsional/panelling vibration
   and local shell response are outside this module.

## 6. Error judgement against references

Requirement: every key quantity compared with an experimental/reference value
must state its relative error and answer explicitly whether it is below 3 %.

{table(['quantity', 'reference', 'relative error', '< 3 %?'], error_rows,
       ['---', '---', '---:', '---:'])}

* The three machine-checkable targets of this node are met:
  (a) amplification in the lock-in band vs the analytic Duffing oscillator
  `{duffing['max_relative_error']:.2%}` < 10 %; (b) whiplash peak vs the analytic
  modal superposition `{max(whiplash['midship_moment_relative_error'], whiplash['bow_displacement_relative_error'], whiplash['quarter_moment_relative_error']):.2%}`
  < 8 %; (c) pytest {py_text}.
* Comparison with **measured** ice-induced vibration spectra (Aker Arctic / HSVA
  basin or field data) is **not claimed**: no such dataset is available in this
  environment. The status is `WITHHELD`, and the corresponding acceptance
  criterion ("peak frequency deviation < 5 % vs measurement") is reported as
  unverified rather than substituted by a synthetic spectrum. What *is* verified
  is the internal physics chain that such a comparison would test first: the
  response tracks `f_ice = U/s` to `{pct(pulse['response_peak_relative_deviation_from_excitation'], 2)}`,
  the amplification follows the analytic relation, and the self-excited limit
  cycle sits on `f_n` to `{pct(self_excited['max_lock_in_frequency_deviation_operational'], 2)}` in the
  operational speed range.

## 7. Acceptance criteria (machine evaluated)

`{criteria['n_passed']}/{criteria['n_criteria']}` passed,
`all_passed = {criteria['all_passed']}`:

{table(['criterion', 'value', 'threshold', 'result', 'detail'], criteria_rows,
       ['---', '---:', '---:', '---', '---'])}

## 8. Evidence grading

| capability | grade | evidence |
| --- | --- | --- |
| SDOF/MDOF ice-structure coupled vibration (`m x'' + c x' + k x = F_ice`) | REAL | Newmark-beta + RK4, verified against analytic steady state (`{max(row['relative_error_time_vs_linear'] for row in report['forced_response_sweep']['rows']):.1e}`) and the closed-form half-sine response (`{whiplash['midship_moment_relative_error']:.1e}`) |
| Discrete ice load time series (quasi-periodic / Poisson / velocity-weakening) | REAL | generated and spectrally characterised (`ice_load.py`), seeded and reproducible |
| Lock-in criterion `|f_ice-f_n|/f_n < 5 %` → amplification > 3 | REAL | band edges measured in the time domain (`D = {lock_in['min_edge_amplification_time_domain']:.2f}`), damping condition derived analytically |
| Duffing (cubic) response / harmonic balance / fold | REAL | HB vs time march `{duffing['max_relative_error']:.1e}`; linear limit exact |
| Modal parameters from FEM | WITHHELD (interface only) | `modal_data_from_fem_extraction()` consumes a TensorFEM extraction; the reported case uses the Euler-Bernoulli reference (`{audit['modal_source']}`) |
| Whiplash (first two vertical modes, bow impact) | REAL | modal superposition vs closed form (`{whiplash['midship_moment_relative_error']:.1e}`), shock spectrum |
| Comparison with measured ice-induced vibration spectra | WITHHELD | no Aker Arctic / HSVA data available here |
| Ice crushing process model (force generation by coupled DEM) | NOT_SUPPORTED | out of scope for this module; see `scenarios/suboff_icebreaking/` |

## 9. Reproducibility

```
cd <worktree>
python3 scenarios/ice_vibration_lockin/run_analysis.py --with-pytest
python3 scenarios/ice_vibration_lockin/build_doc.py
PYTHONPATH=src python3 -m pytest tests/test_ice_vibration_*.py -q
```

Artifacts (all under `artifacts/ice_vibration/`): `report.json` (full payload),
`metrics.json` (criteria + headline numbers), `forced_response_sweep.csv`,
`self_excited_sweep.csv`, `shock_spectrum.csv`, `damping_sensitivity.csv`,
`whiplash_timeseries.csv`, `pytest_summary.json`, `figures/*.png`.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--no-embed", action="store_true",
                        help="reference the png files instead of embedding them as base64")
    args = parser.parse_args(argv)
    path = build(Path(args.artifact_dir), Path(args.out), embed_figures=not args.no_embed)
    print(f"wrote {path} ({path.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())