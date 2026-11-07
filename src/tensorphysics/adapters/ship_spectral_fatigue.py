"""Ship-platform adapter for spectral (frequency-domain) fatigue.

The ship platform (``ship-performance-platform``) owns the wave/ice load side
(``wave_spectrum``, ``ice_load_twin``) and delivers either

1. a tabulated one-sided stress PSD ``{freq_hz: [...], sxx: [...]}`` at the
   hotspot, or
2. a wave/ice spectrum plus the linear transfer function of the structural
   detail (a single-degree-of-freedom resonance, or a tabulated RAO).

This adapter turns both forms into the JSON-ready assessment of
:func:`tensorphysics.fatigue.spectral.spectral_fatigue_assessment`, without
importing anything from the ship platform (the payload is plain JSON, the
boundary stays one-way: ship -> TensorPhysics).

Payload schema (all keys optional except one stress-PSD source)::

    {
      "sn": {"C": 1.0e12, "m": 3.0},              # S^m N = C, stress in MPa
      "stress_psd": {"freq_hz": [...], "sxx": [...]},
      "wave_spectrum": {"type": "jonswap", "hs": 2.0, "tp": 8.0, "gamma": 3.3},
      "transfer": {
          "kind": "sdof",                          # or "rao"
          "fn_hz": 0.5, "damping": 0.02, "gain": 1.0,
          "freq_hz": [...], "amplitude": [...]     # for kind == "rao"
      },
      "bimodal_split_hz": 0.15
    }

The SDOF transfer function is ``|H(f)|^2 = gain^2 / ((1-r^2)^2 +
(2 zeta r)^2)`` with ``r = f/fn``; the stress PSD is ``S_xx = |H|^2 S_wave``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from ..fatigue.spectral import (
    GaussianSpectrum,
    JONSWAP,
    PiersonMoskowitz,
    SpectralMoments,
    moments_from_dict,
    spectral_fatigue_assessment,
    spectral_moments,
)

__all__ = [
    "sdof_transfer",
    "wave_spectrum_from_payload",
    "stress_psd_from_payload",
    "assess_payload",
]


def sdof_transfer(f, fn_hz: float, damping: float, gain: float = 1.0):
    """SDOF compliance transfer ``|H(f)|`` (unitless amplitude ratio)."""
    r = np.asarray(f, dtype=float) / float(fn_hz)
    denom = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * damping * r) ** 2)
    return float(gain) / denom


def wave_spectrum_from_payload(spec: Mapping[str, float]):
    """Build a JONSWAP or Pierson-Moskowitz spectrum object from a payload.

    ``{"type": "jonswap", "hs": .., "tp": .., "gamma": ..}`` (``tp`` is the peak
    period in seconds and is converted to ``fp = 1/tp``) or
    ``{"type": "pm", "hs": .., "tp": ..}``.
    """
    kind = str(spec.get("type", "jonswap")).lower()
    hs = float(spec["hs"])
    if "fp" in spec:
        fp = float(spec["fp"])
    elif "tp" in spec:
        fp = 1.0 / float(spec["tp"])
    else:
        raise ValueError("wave spectrum payload needs 'fp' or 'tp'")
    if kind.startswith("jon"):
        return JONSWAP(hs=hs, fp=fp, gamma=float(spec.get("gamma", 3.3)))
    if kind.startswith("pm") or kind.startswith("pierson"):
        return PiersonMoskowitz(hs=hs, fp=fp)
    raise ValueError(f"unsupported wave spectrum type: {kind!r}")


def stress_psd_from_payload(payload: Mapping) -> Dict[str, object]:
    """Return ``{'freq_hz', 'sxx', 'source'}`` for either payload form."""
    if "stress_psd" in payload:
        block = payload["stress_psd"]
        f = np.asarray(block["freq_hz"], dtype=float)
        s = np.asarray(block["sxx"], dtype=float)
        if f.shape != s.shape or f.size < 5:
            raise ValueError("stress_psd needs matching freq_hz/sxx with >= 5 points")
        order = np.argsort(f)
        return {"freq_hz": f[order], "sxx": s[order], "source": "tabulated"}

    if "wave_spectrum" not in payload:
        raise ValueError("payload needs 'stress_psd' or 'wave_spectrum'")

    spec = wave_spectrum_from_payload(payload["wave_spectrum"])
    transfer = payload.get("transfer") or {"kind": "unit"}
    kind = str(transfer.get("kind", "unit")).lower()

    if kind == "rao":
        f_rao = np.asarray(transfer["freq_hz"], dtype=float)
        a_rao = np.asarray(transfer["amplitude"], dtype=float)
        f = np.union1d(f_rao, np.geomspace(max(f_rao.min(), 1e-4),
                                           f_rao.max(), 4000))
    else:
        hi = float(transfer.get("f_max_hz", 8.0 * spec.fp))
        f = np.geomspace(max(spec.fp / 100.0, 1e-4), hi, 8000)

    s_wave = np.asarray(spec(f), dtype=float)
    if kind == "sdof":
        h = sdof_transfer(f, float(transfer["fn_hz"]),
                          float(transfer.get("damping", 0.02)),
                          float(transfer.get("gain", 1.0)))
    elif kind == "rao":
        h = np.interp(f, f_rao, a_rao, left=0.0, right=0.0)
    elif kind == "unit":
        h = np.ones_like(f)
    else:
        raise ValueError(f"unsupported transfer kind: {kind!r}")
    return {"freq_hz": f, "sxx": s_wave * h ** 2, "source": f"{kind}_transfer",
            "wave_spectrum": spec, "transfer": transfer}


def assess_payload(payload: Mapping) -> Dict[str, object]:
    """Full spectral fatigue assessment from a ship-platform JSON payload.

    Returns the :func:`spectral_fatigue_assessment` dictionary extended with the
    ``source`` of the stress PSD and the wave-spectrum/tranfer description, so
    the caller can store it verbatim.
    """
    sn = payload.get("sn", {"C": 1.0, "m": 3.0})
    psd = stress_psd_from_payload(payload)
    out = spectral_fatigue_assessment(
        psd["freq_hz"], psd["sxx"], C=float(sn["C"]), m=float(sn["m"]),
        bimodal_split_hz=payload.get("bimodal_split_hz"),
    )
    out["stress_psd_source"] = psd["source"]
    out["wave_spectrum"] = None if "wave_spectrum" not in psd else {
        "type": getattr(psd["wave_spectrum"], "name", "unknown"),
        "hs": getattr(psd["wave_spectrum"], "hs", None),
        "fp_hz": getattr(psd["wave_spectrum"], "fp", None),
        "gamma": getattr(psd["wave_spectrum"], "gamma", None),
    }
    out["transfer"] = payload.get("transfer")
    # narrow-band / wide-band applicability verdict for the caller
    a2 = out["moments"]["alpha2"]
    out["applicability"] = {
        "alpha2": a2,
        "narrow_band_valid": bool(a2 > 0.95),
        "dirlik_preferred": bool(0.5 < a2 <= 0.95),
        "note": ("alpha2 > 0.95: all spectral models coincide; 0.5 < alpha2 <= "
                 "0.95: use Dirlik; alpha2 <= 0.5: strongly bimodal, cross-check "
                 "with the combination limit and the time domain"),
    }
    return out