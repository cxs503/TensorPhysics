"""Reference data sets for the adfreeze / breakout model.

Provenance discipline
=====================
Every record carries machine-readable provenance so that a reader (or an
audit script) can tell first-principles content from transcribed literature and
from engineering estimates:

``provenance``
    * ``literature_transcription`` - values transcribed/rounded from published
      experiments or phase diagrams.  ``exact=False`` means "mid-range value of
      a digitised band, not a raw data point"; the ``band`` field gives the
      spread of the published data.
    * ``engineering_estimate`` - representative ship/ice parameters chosen to
      span the design space; not a specific vessel.

These labels are what the audit report keys off: the strength validation is a
*calibration-then-residual* exercise (four temperatures, published bands), the
breakout validation is a *cross-method consistency* check (hull-form
integration vs closed-form rectangular estimate), not independent experimental
validation.
"""

from __future__ import annotations

from .breakout import ShipDefinition

#: Adfreeze shear strength of a freshwater-ice / steel joint, four
#: temperatures.  Mid-range of the published scatter; ``band`` = +-spread.
FREDERKING_ADFREEZE = (
    {
        "T_C": -2.0,
        "tau_MPa": 0.15,
        "band_MPa": 0.05,
        "provenance": "literature_transcription",
        "source": "Frederking (1984); Frederking & Barker (1979) - digitised, mid-range",
        "exact": False,
    },
    {
        "T_C": -5.0,
        "tau_MPa": 0.35,
        "band_MPa": 0.10,
        "provenance": "literature_transcription",
        "source": "Frederking (1984); Frederking & Barker (1979) - digitised, mid-range",
        "exact": False,
    },
    {
        "T_C": -10.0,
        "tau_MPa": 0.70,
        "band_MPa": 0.15,
        "provenance": "literature_transcription",
        "source": "Frederking (1984); Frederking & Barker (1979) - digitised, mid-range",
        "exact": False,
    },
    {
        "T_C": -20.0,
        "tau_MPa": 1.20,
        "band_MPa": 0.25,
        "provenance": "literature_transcription",
        "source": "Frederking (1984); Frederking & Barker (1979) - digitised, mid-range",
        "exact": False,
    },
)

#: Comparative datum used for the salinity penalty: saline ice (S = 32 g/kg)
#: retains about half of the freshwater adfreeze strength at -10 degC.
SALINE_TO_FRESH_RATIO = {
    "T_C": -10.0,
    "salinity_gkg": 32.0,
    "ratio": 0.50,
    "ratio_band": 0.10,
    "provenance": "literature_transcription",
    "source": "adfreeze strength of saline ice, order-of-magnitude datum (Jordaan 2001 review)",
    "exact": False,
}

#: Three vessel classes spanning the icebreaking fleet.  Parameters are
#: representative engineering values (not a named vessel's lines plan).
SHIPS = (
    ShipDefinition(
        name="Baltic 1A-Super escort icebreaker",
        length_pp_m=75.0,
        beam_m=17.5,
        draft_m=6.5,
        ice_thickness_m=0.6,
        beset_length_fraction=0.55,
        bow_entrance_half_angle_deg=22.0,
        midbody_flare_deg=3.0,
        shaft_power_kw=9000.0,
        propeller_diameter_m=3.4,
        propeller_count=2,
        provenance="engineering_estimate",
    ),
    ShipDefinition(
        name="Polar Class 4 medium icebreaker",
        length_pp_m=110.0,
        beam_m=23.0,
        draft_m=8.5,
        ice_thickness_m=1.2,
        beset_length_fraction=0.55,
        bow_entrance_half_angle_deg=20.0,
        midbody_flare_deg=4.0,
        shaft_power_kw=18000.0,
        propeller_diameter_m=4.2,
        propeller_count=2,
        provenance="engineering_estimate",
    ),
    ShipDefinition(
        name="Polar Class 2/3 large polar icebreaker",
        length_pp_m=130.0,
        beam_m=26.0,
        draft_m=9.2,
        ice_thickness_m=1.8,
        beset_length_fraction=0.55,
        bow_entrance_half_angle_deg=18.0,
        midbody_flare_deg=5.0,
        shaft_power_kw=30000.0,
        propeller_diameter_m=4.6,
        propeller_count=2,
        provenance="engineering_estimate",
    ),
)


def frederking_temperatures() -> tuple[float, ...]:
    return tuple(rec["T_C"] for rec in FREDERKING_ADFREEZE)


def frederking_strengths() -> tuple[float, ...]:
    return tuple(rec["tau_MPa"] for rec in FREDERKING_ADFREEZE)