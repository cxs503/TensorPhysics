"""Extreme wave focusing via the nonlinear Schrodinger (NLS) equation and
the Peregrine breather (p472w2-N08).

This package establishes a *third-order* benchmark for extreme wave input in
seakeeping: the one-dimensional focusing NLS (with the Zakharov-reduction
spectral form as the canonical implementation) is solved by a pseudo-spectral
split-step Fourier method, and the Peregrine breather -- the rational
fundamental rogue-wave solution of the NLS -- is used as the exact-solution
anchor.  The package then:

1.  Reproduces the Peregrine breather from the NLS numerical solution
    (phase-aligned L2 relative error <= 2%, C1).
2.  Adjudicates the Benjamin-Feir (modulational) instability growth rate
    against the analytic dispersion relation (<= 3%, C2).
3.  Verifies the three NLS conservation invariants -- mass, momentum, energy --
    over 1e6 steps (drift <= 1e-6, C3).
4.  Quantifies how much the breather peak crest exceeds the *second-order*
    spectral-synthesis crest for the same (H_s, T_p) sea state (ratio +
    uncertainty band, no one-sided verdict, C4).
5.  Applies a linear RAO as a pure transfer function to the breather wave
    field; because a linear transfer function cannot resolve the nonlinear
    crest, this sub-clause is judged NOT_SUPPORTED and a time-domain coupling
    interface contract is delivered instead (C5).
6.  Solves the inverse problem: given a target crest, recover the required
    bandwidth / steepness, then re-synthesise a wave field that reproduces the
    target crest to <= 5% (C6).
7.  Negative control: as the nonlinearity parameter epsilon -> 0 the NLS
    solution degenerates to the linear spectral synthesis (<= 1%, C7).

Physical setup
--------------
The deep-water wave envelope A(x, t) (complex, carrier omega_0 = 1, k_0 = 1)
obeys the standard focusing NLS

    i A_t + (1/2) A_xx + |A|^2 A = 0.

The Peregrine breather on a unit background is the exact rational solution

    A(x, t) = [ 1 - 4(1 + 4 i t) / (1 + 4 x^2 + 16 t^2) ] e^{i t},

whose peak |A| = 3 at (x, t) = (0, 0) is three times the background -- the
canonical rogue-wave amplification.

References
----------
- Peregrine, D.H. (1983). "Water waves, nonlinear Schrodinger equations and
  their solutions." J. Austral. Math. Soc. B 25, 16-43.
- Zakharov, V.E. (1968). "Stability of periodic waves of finite amplitude on
  the surface of a deep fluid." J. Appl. Mech. Tech. Phys. 9, 190-194.
- Benjamin, T.B. & Feir, J.E. (1967). "The disintegration of wave trains on
  deep water." J. Fluid Mech. 27, 417-430.
- Osborne, A.R. (2010). "Nonlinear Ocean Waves and the Inverse Scattering
  Transform." Academic Press.
"""

from .nls_solver import (  # noqa: F401
    NLSSolver,
    NLSResult,
    split_step_fourier_step,
    nls_spectral_solve,
)
from .peregrine import (  # noqa: F401
    PeregrineBreather,
    peregrine_amplitude,
    peregrine_envelope,
    peregrine_peak_amplitude,
    peregrine_phase_aligned_l2_error,
)
from .benjamin_feir import (  # noqa: F401
    benjamin_feir_growth_rate,
    benjamin_feir_dimensionless,
    benjamin_feir_max_growth,
    benjamin_feir_instability_band,
    benjamin_feir_scan_verification,
)
from .conservation import (  # noqa: F401
    nls_mass,
    nls_momentum,
    nls_energy,
    conservation_invariants,
    conservation_drift_study,
)
from .second_order import (  # noqa: F401
    linear_crest,
    second_order_crest,
    second_order_spectral_synthesis,
    breather_vs_second_order,
    crest_ratio_with_uncertainty,
)
from .rao_interface import (  # noqa: F401
    LinearRAO,
    rao_response_amplification,
    rao_peak_response,
    TimeDomainCouplingContract,
    build_time_domain_coupling_contract,
)
from .inverse import (  # noqa: F401
    inverse_bandwidth_from_crest,
    inverse_steepness_from_crest,
    inverse_wave_field,
    inverse_reproduction_error,
    inverse_problem_gate,
)
from .negative_control import (  # noqa: F401
    linear_degeneration_check,
    nls_linear_limit,
    epsilon_linear_spectral_error,
)

__all__ = [
    # NLS solver
    "NLSSolver",
    "NLSResult",
    "split_step_fourier_step",
    "nls_spectral_solve",
    # Peregrine
    "PeregrineBreather",
    "peregrine_amplitude",
    "peregrine_envelope",
    "peregrine_peak_amplitude",
    "peregrine_phase_aligned_l2_error",
    # Benjamin-Feir
    "benjamin_feir_growth_rate",
    "benjamin_feir_dimensionless",
    "benjamin_feir_max_growth",
    "benjamin_feir_instability_band",
    "benjamin_feir_scan_verification",
    # Conservation
    "nls_mass",
    "nls_momentum",
    "nls_energy",
    "conservation_invariants",
    "conservation_drift_study",
    # Second-order
    "linear_crest",
    "second_order_crest",
    "second_order_spectral_synthesis",
    "breather_vs_second_order",
    "crest_ratio_with_uncertainty",
    # RAO interface
    "LinearRAO",
    "rao_response_amplification",
    "rao_peak_response",
    "TimeDomainCouplingContract",
    "build_time_domain_coupling_contract",
    # Inverse
    "inverse_bandwidth_from_crest",
    "inverse_steepness_from_crest",
    "inverse_wave_field",
    "inverse_reproduction_error",
    "inverse_problem_gate",
    # Negative control
    "linear_degeneration_check",
    "nls_linear_limit",
    "epsilon_linear_spectral_error",
]
