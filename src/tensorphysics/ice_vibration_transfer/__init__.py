"""Ice-induced vibration transfer chain: wet modes -> base acceleration -> pipe fatigue.

Campaign node ``p228w1-N08`` of the ice-navigation campaign.  The package
assembles the five links of the chain

``N04 DEM milling torque`` -> ``wet/dry hull modal basis`` -> ``equipment base
acceleration through the mount transfer function H(f)`` -> ``pipe-span stress
spectrum`` -> ``rainflow + DNV-RP-C203 S-N + Miner fatigue life``

and evaluates the node's physical acceptance criteria numerically.

Sub-modules
-----------
``hull_modal``
    Geometry gate (Cb / L / B / displacement), dry free-free Timoshenko modes and
    the wet eigenproblem in the dry basis with the 2-D strip added mass and the
    linearised buoyancy restoring stiffness.
``excitation``
    Ingestion of the N04 DEM milling torque model
    (:mod:`tensorphysics.propeller_fatigue.ice_spectrum`): torque pulse train,
    Welch PSD, and the self-equilibrated hull excitation force couple.
``response``
    Modal-superposition acceleration transfer ``a(f)/F`` and the hull acceleration
    PSD at the equipment foundation station.
``transfer``
    Resilient-mount transmissibility ``H(f)`` (base-excited SDOF), Bode data and
    the band-limited attenuation envelope.
``pipe``
    Simply supported pipe span: modal stress transfer (validated against the
    static ``m L^2/(8Z)`` limit), random-phase stress history synthesis, ASTM
    E1049-85 rainflow counting, DNV-RP-C203 D-curve damage, Miner accumulation and
    the resulting life in years (with a narrowband Rayleigh cross-check).
``chain``
    :func:`chain.run_chain` - the driver producing the full JSON payload with the
    criteria G1/C1-C6, the Bode data and the fatigue-life vs ice-load curve.

The package needs NumPy and is not imported from ``tensorphysics/__init__.py``
so that the core orchestration package keeps its dependency-free import surface.
"""

from __future__ import annotations

from .chain import (
    DEFAULT_SCALE_SWEEP,
    DEFAULT_SEVERITIES,
    GOVERNING_PATH,
    ChainConfig,
    ChainResult,
    replace_scaled,
    run_chain,
)
from .excitation import DESIGN_KQ, IceTorqueExcitation, milling_torque_excitation, welch_psd
from .hull_modal import (
    GEOMETRY_REFERENCE,
    GeometryGate,
    HullModalBasis,
    geometry_gate,
    wet_dry_modal_basis,
)
from .pipe import (
    SECONDS_PER_YEAR,
    PipeSpan,
    fatigue_life,
    narrowband_damage_rate,
    synthesise_stress_history,
)
from .response import (
    ResponseSpectrum,
    acceleration_psd,
    acceleration_transfer,
    band_rms,
    point_force_acceleration,
)
from .mitigation import (
    DEFAULT_WAVE_WHIPPING_LOAD,
    QZSMount,
    band_attenuation,
    design_map,
    select_design,
    service_stroke,
    simulate_base_excitation,
    tolerance_window,
)
from .maturity import MitigationConfig, MitigationResult, run_mitigation
from .transfer import MountIsolation, transmissibility_envelope

__all__ = [
    "DEFAULT_SCALE_SWEEP",
    "DEFAULT_WAVE_WHIPPING_LOAD",
    "MitigationConfig",
    "MitigationResult",
    "QZSMount",
    "DEFAULT_SEVERITIES",
    "GOVERNING_PATH",
    "DESIGN_KQ",
    "GEOMETRY_REFERENCE",
    "SECONDS_PER_YEAR",
    "ChainConfig",
    "ChainResult",
    "GeometryGate",
    "HullModalBasis",
    "IceTorqueExcitation",
    "MountIsolation",
    "PipeSpan",
    "ResponseSpectrum",
    "acceleration_psd",
    "acceleration_transfer",
    "band_rms",
    "fatigue_life",
    "geometry_gate",
    "milling_torque_excitation",
    "band_attenuation",
    "design_map",
    "run_mitigation",
    "select_design",
    "service_stroke",
    "simulate_base_excitation",
    "tolerance_window",
    "narrowband_damage_rate",
    "point_force_acceleration",
    "replace_scaled",
    "run_chain",
    "synthesise_stress_history",
    "transmissibility_envelope",
    "welch_psd",
    "wet_dry_modal_basis",
]