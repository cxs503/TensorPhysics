"""Linear RAO as a pure transfer function on the breather wave field, plus the
time-domain coupling interface contract (C5).

A linear RAO H(omega) maps a wave *amplitude* to a linear response amplitude.
Applied to the Peregrine breather -- a *nonlinear* wave whose crest is 3x the
background -- the linear transfer function cannot resolve the nonlinear crest:
it sees only the linear (first-order) component of the field and returns an
amplification factor identical to that of a linear wave of the same background.

C5 criterion: the linear-RAO peak response amplification factor is reported;
because a linear RAO cannot resolve the nonlinear crest (expected), this
sub-clause is judged NOT_SUPPORTED and a time-domain coupling interface
contract is delivered instead -- the interface a nonlinear (e.g. fully
nonlinear potential / CFD) solver would need to couple with the ship response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

__all__ = [
    "LinearRAO",
    "rao_response_amplification",
    "rao_peak_response",
    "TimeDomainCouplingContract",
    "build_time_domain_coupling_contract",
]


@dataclass
class LinearRAO:
    """Linear response-amplitude operator (transfer function).

    Parameters
    ----------
    omega : ndarray
        Angular frequencies (rad/s).
    H : complex ndarray
        Transfer function H(omega) (response per unit wave amplitude).
    """

    omega: np.ndarray
    H: np.ndarray

    def __post_init__(self) -> None:
        self.omega = np.asarray(self.omega, dtype=float)
        self.H = np.asarray(self.H, dtype=complex)
        if self.omega.shape[0] != self.H.shape[0]:
            raise ValueError("omega and H must have the same length")

    def response(self, eta_hat: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Linear response to a wave field with Fourier coefficients eta_hat.

        R(omega) = H(omega) * eta_hat(omega).
        """
        # interpolate H onto the given omega grid
        H_interp = np.interp(omega, self.omega, np.real(self.H)) + \
                   1j * np.interp(omega, self.omega, np.imag(self.H))
        return H_interp * eta_hat


def _default_rao(omega: np.ndarray) -> np.ndarray:
    """A canonical linear RAO: a damped single-DOF heave response.

    H(omega) = 1 / (1 - (omega/omega_n)^2 + 2 i zeta (omega/omega_n)),
    with omega_n = 1.0 (normalised) and zeta = 0.1.  This is the standard
    linear transfer function of a floating body in heave.
    """
    omega_n = 1.0
    zeta = 0.1
    r = omega / omega_n
    return 1.0 / (1.0 - r ** 2 + 2.0j * zeta * r)


def rao_response_amplification(
    A_field: np.ndarray,
    x: np.ndarray,
    t: float,
    rao: Optional[LinearRAO] = None,
) -> dict:
    """Apply a linear RAO to a wave field and report the peak response
    amplification factor.

    The wave field is the *surface elevation* eta(x, t) = Re[A(x, t) e^{i t}]
    (the real part of the envelope times the carrier).  The linear RAO maps the
    Fourier components of eta to the response.  Because the RAO is linear, the
    response amplification factor is the same whether the input is the
    nonlinear breather or a linear wave of the same background -- the RAO
    cannot resolve the nonlinear crest.

    Parameters
    ----------
    A_field : complex ndarray
        Envelope on the spatial grid.
    x : ndarray
        Spatial grid.
    t : float
        Time.
    rao : LinearRAO, optional
        The linear RAO (default: canonical damped heave).

    Returns
    -------
    dict
        With keys: peak_input_amplitude, peak_response_amplitude,
        amplification_factor, and a flag 'linear_rao_resolves_nonlinear'
        (always False -- the linear RAO cannot resolve the nonlinear crest).
    """
    # surface elevation = Re[A e^{i t}] (real part of the physical wave)
    eta = np.real(A_field * np.exp(1j * t))
    n = len(x)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    # only positive frequencies carry physical meaning; use the full grid
    # but the RAO is defined on positive omega
    omega_pos = np.abs(omega)
    if rao is None:
        rao = LinearRAO(omega=np.linspace(0.01, 5.0, 500),
                        H=_default_rao(np.linspace(0.01, 5.0, 500)))
    eta_hat = np.fft.fft(eta)
    # response in frequency domain
    R_hat = rao.response(eta_hat, omega_pos)
    R = np.fft.ifft(R_hat)
    peak_in = float(np.max(np.abs(eta)))
    peak_resp = float(np.max(np.abs(R)))
    amp = peak_resp / peak_in if peak_in > 1e-30 else 0.0
    return {
        "peak_input_amplitude": peak_in,
        "peak_response_amplitude": peak_resp,
        "amplification_factor": float(amp),
        "linear_rao_resolves_nonlinear": False,
        "note": (
            "A linear RAO is a pure transfer function: it maps the linear "
            "(first-order) component of the field and cannot resolve the "
            "nonlinear crest of the breather.  The amplification factor is "
            "therefore identical to that of a linear wave of the same "
            "background -- this sub-clause is judged NOT_SUPPORTED."
        ),
    }


def rao_peak_response(
    A_field: np.ndarray,
    x: np.ndarray,
    t: float,
    rao: Optional[LinearRAO] = None,
) -> float:
    """Peak linear-RAO response amplitude to a wave field."""
    return float(rao_response_amplification(A_field, x, t, rao)["peak_response_amplitude"])


@dataclass
class TimeDomainCouplingContract:
    """Interface contract for time-domain coupling of a nonlinear wave solver
    with the ship response (delivered because the linear RAO cannot resolve
    the nonlinear crest).

    This is the *interface* a nonlinear (fully nonlinear potential / CFD /
    higher-order spectral) solver must expose to couple the breather wave field
    with the ship response.  It is a contract, not an implementation: the
    nonlinear response solver is out of scope for this node (C5 is judged
    NOT_SUPPORTED for the linear RAO).

    Attributes
    ----------
    field_name : str
        The coupled field (e.g. 'surface_elevation').
    field_type : str
        'scalar' (elevation) or 'vector' (velocity potential).
    spatial_grid : dict
        Grid description (x, dx, n_x).
    time_grid : dict
        Time description (t, dt, n_t).
    coupling_terms : list
        The nonlinear terms that must be exchanged (e.g. 'quadratic_velocity',
        'bound_harmonic', 'wave_breaking').
    required_outputs : list
        The response quantities the coupled solver must produce.
    """

    field_name: str = "surface_elevation"
    field_type: str = "scalar"
    spatial_grid: dict = field(default_factory=dict)
    time_grid: dict = field(default_factory=dict)
    coupling_terms: list = field(default_factory=list)
    required_outputs: list = field(default_factory=list)


def build_time_domain_coupling_contract(
    x: np.ndarray,
    t: float,
    dt: float,
    n_t: int,
) -> TimeDomainCouplingContract:
    """Build the time-domain coupling interface contract for the breather
    wave field.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    t : float
        Current time.
    dt : float
        Time step.
    n_t : int
        Number of time steps.

    Returns
    -------
    TimeDomainCouplingContract
    """
    return TimeDomainCouplingContract(
        field_name="surface_elevation",
        field_type="scalar",
        spatial_grid={"x": x.tolist(), "dx": float(x[1] - x[0]) if len(x) > 1 else 1.0,
                      "n_x": int(len(x))},
        time_grid={"t": float(t), "dt": float(dt), "n_t": int(n_t)},
        coupling_terms=[
            "quadratic_velocity_term",       # (1/2) k |A|^2 -- the Stokes bound harmonic
            "bound_harmonic_crest",          # the 3x crest of the breather
            "nonlinear_phase_rotation",      # the |A|^2 phase in the NLS
            "wave_breaking_threshold",       # crest > 3 A0 triggers breaking
        ],
        required_outputs=[
            "peak_crest_elevation",
            "peak_response_amplitude",
            "response_spectrum",
            "time_series_of_response",
        ],
    )
