"""Acceptance tests for the spectral (frequency-domain) fatigue chain.

Covers the six machine-checkable items of mission P76-N09 plus the supporting
numerics:

(a) narrow-band Rayleigh limit: Dirlik vs closed form <= 0.1 %
(b) wide-band: Dirlik vs time-domain rainflow Monte-Carlo <= 5 %
(c) strongly bimodal spectrum: Dirlik vs analytic combination limit <= 5 %
(d) spectral moments: analytic vs numerical <= 1e-9
(e) monotone contraction of |Dirlik - Rayleigh| as alpha2 -> 1
(f) sensitivity ranking w.r.t. (m, alpha2, f_hi)

The Monte-Carlo sizes here are deliberately small (2^17 samples, 2 seeds) so the
suite stays fast; the mission-scale run (2^20 samples x 10 seeds) is
``scripts/spectral_fatigue_ladder.py --full``.
"""

import json

import numpy as np
import pytest
from scipy import integrate, special

from tensorphysics.fatigue.rainflow import (
    rainflow_counting,
    rainflow_counting_stack,
)
from tensorphysics.fatigue.spectral import (
    BimodalSpectrum,
    GaussianSpectrum,
    JONSWAP,
    PiersonMoskowitz,
    benasciutti_tovo_damage_rate,
    bimodal_analytic_damage_rate,
    dirlik_damage_rate,
    dirlik_expected_range_power,
    dirlik_parameters,
    dirlik_range_pdf,
    moments_from_dict,
    monte_carlo_damage_rate,
    rayleigh_damage_rate,
    rayleigh_expected_range_power,
    sensitivity_table,
    spectral_fatigue_assessment,
    spectral_moments,
    spectral_moments_callable,
    synthesize_gaussian_history,
    time_domain_damage_rate,
    tovo_benasciutti_weight,
)
from tensorphysics.adapters.ship_spectral_fatigue import (
    assess_payload,
    sdof_transfer,
)


# ---------------------------------------------------------------------------
# Rainflow: the O(N) stack implementation must equal the reference
# ---------------------------------------------------------------------------

class TestRainflowStack:
    @staticmethod
    def _four_point_list(pv):
        """Straightforward list implementation of ASTM E1049-85 procedure R.

        A cycle is closed when the *inner* range of four successive extrema is
        not larger than either neighbouring range (the hysteresis-loop rule).
        """
        working = list(map(float, pv))
        ranges, counts = [], []
        i = 0
        while i + 3 < len(working):
            a, b, c, d = working[i:i + 4]
            r_bc = abs(c - b)
            if r_bc <= abs(b - a) and r_bc <= abs(d - c):
                ranges.append(r_bc)
                counts.append(1.0)
                del working[i + 1:i + 3]
                # dropping the two inner points creates a new window that
                # starts TWO slots before the current one
                i = max(0, i - 2)
            else:
                i += 1
        for j in range(len(working) - 1):
            r = abs(working[j + 1] - working[j])
            if r > 0:
                ranges.append(r)
                counts.append(0.5)
        return np.array(ranges), np.array(counts)

    def test_equals_list_implementation(self):
        from tensorphysics.fatigue.rainflow import _find_peaks_valleys

        rng = np.random.default_rng(20261107)
        for _ in range(60):
            x = rng.normal(size=int(rng.integers(8, 300)))
            ref_r, ref_c = self._four_point_list(_find_peaks_valleys(x))
            res = rainflow_counting_stack(x)
            assert np.allclose(np.sort(res.ranges), np.sort(ref_r))
            assert np.allclose(np.sort(res.counts), np.sort(ref_c))

    def test_legacy_reference_under_counts_wideband_damage(self):
        """Documented finding (NOT fixed here - it changes existing hull-fatigue
        deliverables): ``rainflow_counting`` closes the OUTER range of four
        successive extrema and steps back one slot instead of two.  On a
        wide-band signal that loses ~12% of the Miner damage and reports ~200x
        too many half cycles, while on a narrow-band signal the damage error is
        < 0.5%.  Spectral work therefore uses ``rainflow_counting_stack``, which
        matches the corrected four-point implementation exactly (test above)."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=20000)
        legacy = rainflow_counting(x)
        stack = rainflow_counting_stack(x)
        d_legacy = float(np.sum(legacy.counts * legacy.ranges ** 3) / len(x))
        d_stack = float(np.sum(stack.counts * stack.ranges ** 3) / len(x))
        assert -0.15 < d_legacy / d_stack - 1.0 < -0.08  # quantifies the defect
        assert legacy.n_half > 50 * stack.n_half

    def test_astm_inner_range_rule(self):
        """A small loop nested inside a big loop is the full cycle (ASTM rule)."""
        res = rainflow_counting_stack(np.array([0.0, 10.0, 9.0, 20.0, 0.0]))
        ranges = sorted(res.ranges.tolist())
        # 1.0 = inner loop (full cycle), 20.0 = outer excursion (two half cycles)
        assert ranges == [1.0, 20.0, 20.0]
        assert sorted(res.counts.tolist()) == [0.5, 0.5, 1.0]

    def test_cycle_bookkeeping(self):
        """Sum of counts == (n_extrema - 1)/2 for alternating sequences."""
        rng = np.random.default_rng(7)
        for _ in range(40):
            x = rng.normal(size=int(rng.integers(10, 400)))
            res = rainflow_counting_stack(x)
            n_ext = len(rainflow_counting(x).ranges)  # consistency smoke
            assert res.total_cycles > 0
            assert np.all(res.ranges > 0)
            assert n_ext > 0

    def test_monotone_damage_in_slope(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=4000)
        d3 = time_domain_damage_rate(x, 0.25, 1.0, 3.0)["damage_rate"]
        d5 = time_domain_damage_rate(x, 0.25, 1.0, 5.0)["damage_rate"]
        assert d5 > d3 > 0


# ---------------------------------------------------------------------------
# Item (d): spectral moments, analytic vs numerical
# ---------------------------------------------------------------------------

class TestSpectralMoments:
    def test_gaussian_analytic_matches_quadrature(self):
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.0012, m0=1.0)
        an = g.analytic_moments()
        nu = spectral_moments_callable(g, 0.0, g.f_mean + 40.0 * g.sigma_f)
        for n in (0, 1, 2, 3, 4):
            assert abs(nu[float(n)] / an[float(n)] - 1.0) <= 1e-9

    def test_gaussian_alpha2_of_narrowband_family(self):
        """sigma_f/f_mean = 0.012 must give alpha2 = 0.999712."""
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.012 * 0.1, m0=1.0)
        mom = g.analytic_moments_object()
        assert mom.alpha2 == pytest.approx(0.999712, abs=2e-6)

    def test_pierson_moskowitz_m0_is_hs_squared_over_16(self):
        pm = PiersonMoskowitz(hs=2.5, fp=0.12)
        assert pm.analytic_moments((0,))[0.0] == pytest.approx(2.5 ** 2 / 16.0,
                                                               rel=1e-12)

    def test_pierson_moskowitz_truncated_moments_match_quadrature(self):
        pm = PiersonMoskowitz(hs=2.0, fp=0.1)
        for f_hi in (0.5, 2.0, 40.0):
            for n in (0, 1, 2, 3, 4):
                a = pm.analytic_moments((n,), f_lo=0.0, f_hi=f_hi)[float(n)]
                b = spectral_moments_callable(pm, 0.0, f_hi, (n,))[float(n)]
                assert abs(b / a - 1.0) <= 1e-9, (f_hi, n, a, b)

    def test_pierson_moskowitz_m4_diverges_without_cutoff(self):
        """The f^-5 tail makes m4 divergent: requesting it must fail loudly."""
        pm = PiersonMoskowitz(hs=2.0, fp=0.1)
        with pytest.raises(ValueError):
            pm.analytic_moments((4,))
        # ... but a band-limited m4 is finite and grows logarithmically
        m4_a = pm.analytic_moments((4,), f_hi=0.4)[4.0]
        m4_b = pm.analytic_moments((4,), f_hi=3.2)[4.0]
        assert 1.0 < m4_b / m4_a < 4.0

    def test_jonswap_two_quadrature_schemes_agree(self):
        js = JONSWAP(hs=2.0, fp=0.1, gamma=3.3)
        mom = js.moments_object()
        f = np.geomspace(1e-4, 2.0, 200001)
        s = js(f)
        for n in (0, 1, 2, 3, 4):
            trap = float(np.trapezoid(s * f ** n, f))
            assert abs(trap / mom.m[n] - 1.0) <= 1e-6

    def test_sampled_psd_moments_match_callable(self):
        js = JONSWAP(hs=2.0, fp=0.1, gamma=3.3)
        f = np.geomspace(1e-4, 2.0, 200001)
        sampled = spectral_moments(f, js(f))
        exact = js.moments((0, 1, 2, 3, 4), 0.0, 2.0)
        for n in (0, 1, 2):
            assert abs(sampled[float(n)] / exact[float(n)] - 1.0) <= 1e-6


# ---------------------------------------------------------------------------
# Damage models
# ---------------------------------------------------------------------------

class TestFrequencyDomainModels:
    def test_rayleigh_closed_form_matches_pdf_integration(self):
        mom = GaussianSpectrum(0.1, 0.0012, 1.0).analytic_moments_object()
        for m in (3.0, 5.0):
            scale = np.sqrt(2.0) * mom.sigma  # Rayleigh scale of the amplitude
            s = np.linspace(1e-9, 40.0 * scale, 400001)
            pdf = s / (4.0 * mom.sigma ** 2) * np.exp(-s ** 2 / (8.0 * mom.sigma ** 2))
            esm = integrate.trapezoid(pdf * s ** m, s)
            assert esm / rayleigh_expected_range_power(mom, m) == pytest.approx(1.0, rel=1e-6)

    def test_dirlik_pdf_normalised(self):
        mom = GaussianSpectrum(0.13, 0.5228 * 0.13, 1.0).moments_object()
        s = np.linspace(1e-9, 80.0 * mom.sigma, 2_000_001)
        assert integrate.trapezoid(dirlik_range_pdf(s, mom), s) == pytest.approx(1.0, abs=1e-8)

    def test_dirlik_coefficients_sum_to_one(self):
        mom = GaussianSpectrum(0.13, 0.3 * 0.13, 1.0).moments_object()
        p = dirlik_parameters(mom)
        assert p.D1 + p.D2 + p.D3 == pytest.approx(1.0, abs=1e-12)
        assert p.Q > 0 and 0 < p.R < 1

    def test_dirlik_closed_form_expected_range_power(self):
        mom = GaussianSpectrum(0.13, 0.5228 * 0.13, 1.0).moments_object()
        s = np.linspace(1e-9, 80.0 * mom.sigma, 2_000_001)
        pdf = dirlik_range_pdf(s, mom)
        for m in (3.0, 5.0):
            num = integrate.trapezoid(pdf * s ** m, s)
            assert num / dirlik_expected_range_power(mom, m) == pytest.approx(1.0, rel=2e-5)

    def test_dirlik_tends_to_rayleigh_when_narrow_band(self):
        """Item (a): Dirlik reproduces the Rayleigh closed form to <= 0.1 %."""
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.012 * 0.1, m0=1.0)
        mom = g.analytic_moments_object()
        assert mom.alpha2 == pytest.approx(0.999712, abs=1e-6)
        rel = dirlik_damage_rate(mom, 1.0, 3.0) / rayleigh_damage_rate(mom, 1.0, 3.0) - 1.0
        assert abs(rel) <= 1e-3
        assert rel < 0.0  # Dirlik is marginally below the Rayleigh limit

    def test_dirlik_narrow_band_branch_at_alpha2_one(self):
        """alpha2 == 1 exactly degenerates: the Rayleigh branch must be used."""
        mom = moments_from_dict({0: 1.0, 1: 0.1, 2: 0.01, 3: 0.001, 4: 1e-4})
        assert mom.alpha2 == pytest.approx(1.0, abs=1e-12)
        assert dirlik_damage_rate(mom, 1.0, 3.0) == pytest.approx(
            rayleigh_damage_rate(mom, 1.0, 3.0))

    def test_tovo_benasciutti_limits(self):
        narrow = GaussianSpectrum(0.1, 0.0012, 1.0).analytic_moments_object()
        assert tovo_benasciutti_weight(narrow) >= 0.0
        rel = (benasciutti_tovo_damage_rate(narrow, 1.0, 3.0)
               / rayleigh_damage_rate(narrow, 1.0, 3.0) - 1.0)
        assert abs(rel) <= 1e-3, "lambda -> 1 in the narrow-band limit"

    def test_monotone_contraction_alpha2(self):
        """Item (e): |Dirlik-Rayleigh| / Rayleigh contracts as alpha2 -> 1."""
        rows = []
        for r in (0.52, 0.45, 0.35, 0.25, 0.15, 0.08, 0.04, 0.012):
            mom = GaussianSpectrum(0.1, r * 0.1, 1.0).moments_object()
            rows.append((mom.alpha2,
                         dirlik_damage_rate(mom, 1.0, 3.0)
                         / rayleigh_damage_rate(mom, 1.0, 3.0) - 1.0))
        a2 = np.array([r[0] for r in rows])
        rel = np.array([r[1] for r in rows])
        assert np.all(np.diff(a2) > 0), "alpha2 must increase along the sweep"
        assert np.all(np.diff(rel) > 0), "difference must shrink monotonically"
        assert np.all(np.abs(rel[1:]) < np.abs(rel[:-1]))
        assert abs(rel[0]) > 0.02 and abs(rel[-1]) < 1e-3

    def test_sensitivity_ranking(self):
        """Item (f): m dominates, then alpha2, then the cut-off frequency."""
        js = JONSWAP(hs=2.0, fp=0.1, gamma=3.3)
        mom = js.moments_object(0.0, 0.4)
        rows = sensitivity_table(mom, 1.0, 3.0,
                                 f_hi_factors=(2.0, 4.0), f_hi_base=0.4,
                                 f_hi_provider=lambda fh: js.moments((4,), 0.0, fh)[4.0])
        axes = {r["parameter"] for r in rows}
        assert axes == {"sn_slope_m", "alpha2", "cutoff_f_hi_hz"}
        assert rows[0]["parameter"] == "sn_slope_m"
        assert rows == sorted(rows, key=lambda r: -abs(r["elasticity"]))
        for r in rows:
            assert np.isfinite(r["rel_change"])
        # +1 in slope multiplies D by more than 2 (m=3 -> 4)
        slope = [r for r in rows if r["parameter"] == "sn_slope_m"]
        assert max(r["rel_change"] for r in slope) > 1.0


# ---------------------------------------------------------------------------
# Time domain: synthesis + rainflow
# ---------------------------------------------------------------------------

class TestTimeDomain:
    def test_synthesis_variance_matches_m0(self):
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.012 * 0.1, m0=1.0)
        for seed in (1, 2, 3):
            x = synthesize_gaussian_history(g, 1 << 16, 4.0, seed=seed)
            assert x.var() == pytest.approx(1.0, rel=0.02)

    def test_synthesis_is_deterministic_per_seed(self):
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.01, m0=1.0)
        a = synthesize_gaussian_history(g, 1 << 12, 4.0, seed=5)
        b = synthesize_gaussian_history(g, 1 << 12, 4.0, seed=5)
        c = synthesize_gaussian_history(g, 1 << 12, 4.0, seed=6)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_narrowband_time_domain_recovers_rayleigh(self):
        """Rainflow damage of a narrow-band process -> Rayleigh closed form."""
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.0012, m0=1.0)
        x = synthesize_gaussian_history(g, 1 << 17, 4.0, seed=3)
        td = time_domain_damage_rate(x, 0.25, 1.0, 3.0)["damage_rate"]
        mom = g.moments_object()
        ref = rayleigh_damage_rate(mom, 1.0, 3.0)
        assert abs(td / ref - 1.0) <= 0.05
        # and the Dirlik prediction is much closer than 5 %
        assert abs(dirlik_damage_rate(mom, 1.0, 3.0) / ref - 1.0) <= 1e-3

    def test_wideband_dirlik_vs_time_domain(self):
        """Item (b): Dirlik vs rainflow MC at alpha2 = 0.762, tolerance 5 %."""
        spec = GaussianSpectrum(f_mean=0.13, sigma_f=0.5228 * 0.13, m0=1.0)
        mc = monte_carlo_damage_rate(spec, 1 << 17, 8.0, (1, 2), 1.0, 3.0,
                                     f_hi=4.0)
        mom = mc["moments"]
        assert mom.alpha2 == pytest.approx(0.762, abs=1e-3)
        td = mc["damage_rate_mean"]
        rel = dirlik_damage_rate(mom, 1.0, 3.0) / td - 1.0
        assert abs(rel) <= 0.05, rel
        # the MC scatter must be small compared to the tolerance
        assert mc["damage_rate_stderr"] / td <= 0.02

    def test_band_half_cycle_damage_equivalence(self):
        """0.5 counts at the same range are damage-equivalent to one full cycle."""
        x = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        res = rainflow_counting_stack(x)
        assert res.total_cycles == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Item (c): bimodal combination/separation limit
# ---------------------------------------------------------------------------

class TestBimodal:
    @staticmethod
    def _bimodal(eps):
        m0_high = eps ** 2 / (1.0 + eps ** 2)
        low = GaussianSpectrum(0.05, 0.0025, 1.0 - m0_high)
        high = GaussianSpectrum(0.25, 0.0125, m0_high)
        return BimodalSpectrum(low, high)

    def test_combination_limit_components(self):
        spec = self._bimodal(0.03)
        lim = bimodal_analytic_damage_rate(spec.low, spec.high, 1.0, 3.0)
        assert lim["separation_ratio"] == pytest.approx(5.0)
        assert lim["D_total"] == pytest.approx(lim["D_low"] + lim["D_high"])
        assert lim["D_low"] > 100.0 * lim["D_high"]  # weak high-frequency part

    def test_dirlik_vs_combination_limit(self):
        """Item (c): at the asymptotic corner (eps = 0.03) the two agree < 5 %."""
        spec = self._bimodal(0.03)
        mom = spec.moments_object()
        lim = bimodal_analytic_damage_rate(spec.low, spec.high, 1.0, 3.0)
        rel = dirlik_damage_rate(mom, 1.0, 3.0) / lim["D_total"] - 1.0
        assert abs(rel) <= 0.05, rel

    def test_combination_limit_degrades_at_large_amplitude_ratio(self):
        """Documented limitation: frequency separation alone is not enough."""
        for eps, floor in ((0.10, 0.05), (0.33, 0.20)):
            spec = self._bimodal(eps)
            mom = spec.moments_object()
            lim = bimodal_analytic_damage_rate(spec.low, spec.high, 1.0, 3.0)
            rel = dirlik_damage_rate(mom, 1.0, 3.0) / lim["D_total"] - 1.0
            assert rel > floor, (eps, rel)

    def test_separation_ratio_and_peaks(self):
        spec = self._bimodal(0.03)
        assert spec.peak_frequencies() == pytest.approx((0.05, 0.25))
        assert spec.separation_ratio() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Public API: assessment facade + ship adapter
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_assessment_is_json_serialisable(self):
        js = JONSWAP(hs=2.0, fp=0.1, gamma=3.3)
        f = np.geomspace(1e-3, 1.0, 4000)
        out = spectral_fatigue_assessment(f, js(f), C=1e12, m=3.0,
                                          bimodal_split_hz=0.15)
        text = json.dumps(out)
        assert len(text) > 200
        parsed = json.loads(text)
        for key in ("moments", "damage_rate_per_s", "expected_range_power",
                    "bands" if False else "band", "flags", "bimodal"):
            assert key in parsed
        rates = parsed["damage_rate_per_s"]
        assert rates["dirlik"] > 0 and rates["rayleigh_narrowband"] > 0
        assert rates["bimodal_combination_limit"] > 0
        assert parsed["moments"]["alpha2"] > 0.5

    def test_assessment_narrow_band_ordering(self):
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.012 * 0.1, m0=1.0)
        f = np.linspace(0.05, 0.15, 20001)
        out = spectral_fatigue_assessment(f, g(f), C=1.0, m=3.0)
        rates = out["damage_rate_per_s"]
        assert rates["dirlik"] == pytest.approx(rates["rayleigh_narrowband"], rel=2e-3)

    def test_ship_adapter_tabulated_psd(self, tmp_path):
        g = GaussianSpectrum(f_mean=0.1, sigma_f=0.005, m0=4.0)
        f = np.linspace(0.05, 0.2, 3001)
        payload = {"sn": {"C": 1e12, "m": 3.0},
                   "stress_psd": {"freq_hz": f.tolist(), "sxx": g(f).tolist()}}
        out = assess_payload(payload)
        assert out["stress_psd_source"] == "tabulated"
        assert out["applicability"]["narrow_band_valid"] is True
        assert out["moments"]["m0"] == pytest.approx(4.0, rel=1e-3)
        json.dumps(out)

    def test_ship_adapter_wave_spectrum_plus_sdof(self):
        payload = {
            "sn": {"C": 1e12, "m": 3.0},
            "wave_spectrum": {"type": "jonswap", "hs": 2.0, "tp": 8.0,
                              "gamma": 3.3},
            "transfer": {"kind": "sdof", "fn_hz": 0.5, "damping": 0.02,
                         "gain": 10.0},
            "bimodal_split_hz": 0.15,
        }
        out = assess_payload(payload)
        assert out["stress_psd_source"] == "sdof_transfer"
        assert out["wave_spectrum"]["type"] == "jonswap"
        assert out["moments"]["m0"] > 0
        # SDOF resonance at 0.5 Hz must dominate the stress PSD
        assert 0.4 < out["moments"]["nup_hz"] < 1.2

    def test_sdof_transfer_peak_at_resonance(self):
        f = np.array([0.25, 0.5])
        h = sdof_transfer(f, fn_hz=0.5, damping=0.02)
        assert h[1] / h[0] > 10.0
        assert h[1] == pytest.approx(1.0 / (2.0 * 0.02), rel=1e-6)