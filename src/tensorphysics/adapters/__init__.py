"""Optional adapters for specialist Tensor solvers.

Import adapters from this package explicitly; the orchestration core never
imports solver packages as a side effect.
"""

from .tensordem import TensorDEMAdapter
from .ship_spectral_fatigue import (
    assess_payload,
    sdof_transfer,
    stress_psd_from_payload,
    wave_spectrum_from_payload,
)

__all__ = [
    "TensorDEMAdapter",
    "assess_payload",
    "sdof_transfer",
    "stress_psd_from_payload",
    "wave_spectrum_from_payload",
]
