"""Ice-excited hull vibration–fatigue–acoustics–cargo-shift multi-physics chain.

Integrated assessment chain for ice-going vessels (P519-N10):

1. **Vibration** — ice-load excitation → hull frequency-domain response →
   1/3-octave acceleration spectrum + ISO 19906 envelope.
2. **Fatigue** — Miner linear cumulative damage + S-N curve + rainflow counting
   (wraps existing ``tensorphysics.fatigue`` and ``tensorphysics.ice_fatigue``).
3. **Acoustics** — ice-breaking energy → source spectrum → Thorp propagation
   attenuation → 1/3-octave SPL → marine-mammal threshold comparison.
4. **Cargo shift** — ice-zone roll/pitch acceleration → cargo sliding/overturning
   per IMO CSS Code Annex 13.

Verification targets
--------------------
- Zero-load ⇒ acceleration = 0
- 1/3-octave spectrum energy conservation error < 1 %
- Zero stress amplitude ⇒ Miner damage = 0
- Miner D ≤ 1 ⇒ safe
- E_break = 0 ⇒ SL = −∞ ⇒ RL = −∞
- Thorp α(10 kHz) ≈ 1 dB/km (error < 10 %)
- Zero acceleration ⇒ cargo slide = 0
- Dimensional consistency check passes
- Deterministic: same input ⇒ same SHA-256

References
----------
- ISO 19906 (2019): Arctic offshore structures — ice actions
- DNV-OS-C401: Fatigue design of offshore steel structures
- ASTM E1049-85: Rainflow cycle counting
- IMO CSS Code Annex 13: Cargo stowage and securing
- Thorp (1967): Analytic description of ambient noise in the sea
- Southall et al. (2007): Marine mammal noise exposure criteria
"""

from .vibration import (
    OneThirdOctaveBands,
    ISO19906Envelope,
    compute_one_third_octave_spectrum,
    compute_iso19906_envelope,
    acceleration_rms_by_band,
    energy_conservation_error,
)
from .acoustics import (
    ThorpAttenuation,
    SourceLevel,
    ReceivedLevel,
    MarineMammalThresholds,
    thorp_attenuation_db_per_km,
    source_level_from_breaking_energy,
    received_level_at_range,
    one_third_octave_spl,
    marine_mammal_exceedance,
)
from .cargo_shift import (
    CargoShiftCriteria,
    CargoShiftResult,
    imo_css_sliding_check,
    imo_css_overturning_check,
    transverse_acceleration,
    longitudinal_acceleration,
)
from .chain import (
    IceMultiPhysicsChain,
    IceMultiPhysicsResult,
    run_ice_multiphysics_assessment,
)

__all__ = [
    # Vibration
    "OneThirdOctaveBands",
    "ISO19906Envelope",
    "compute_one_third_octave_spectrum",
    "compute_iso19906_envelope",
    "acceleration_rms_by_band",
    "energy_conservation_error",
    # Acoustics
    "ThorpAttenuation",
    "SourceLevel",
    "ReceivedLevel",
    "MarineMammalThresholds",
    "thorp_attenuation_db_per_km",
    "source_level_from_breaking_energy",
    "received_level_at_range",
    "one_third_octave_spl",
    "marine_mammal_exceedance",
    # Cargo shift
    "CargoShiftCriteria",
    "CargoShiftResult",
    "imo_css_sliding_check",
    "imo_css_overturning_check",
    "transverse_acceleration",
    "longitudinal_acceleration",
    # Chain
    "IceMultiPhysicsChain",
    "IceMultiPhysicsResult",
    "run_ice_multiphysics_assessment",
]
