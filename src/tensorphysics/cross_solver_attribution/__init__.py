"""Cross-solver arbitration and term-level error attribution (campaign node P78-N08).

The package answers one question: *when two discretisations of the same equation
disagree, what is the disagreement made of?*  It does that with a same-solution
driving protocol (same grid, same CFL, same analytic data), a 2^3 elimination
design over three protocol factors (truncation order ``T``, boundary closure
``B``, limiter/sub-cycling ``L``) and a Shapley/ANOVA decomposition of the
measured disagreement, plus two non-DG/FVM legs (SPH vs LBM) for the viscous and
shallow-water anchors.

Entry points
------------
``make_legs`` / ``factorial_ledger``
    the DG/FVM arbitration legs and their term-level ledgers.
``order_table`` / ``ritter_convergence``
    observed convergence orders for every solver variant.
``legacy_replay``
    the item-level account of the upstream 24.9 % headline.
``lbm_poiseuille`` / ``sph_poiseuille`` / ``lbm_linear_wave`` / ``sph_shallow_wave``
    the SPH and LBM anchor legs.
``cli.main``
    one-click driver writing ``artifacts/cross_solver_attribution/ledger.json``.
"""
from .cases import (LinearWaveCase, ScalarCase, SWECase, advection_sine,
                    cell_averages, diffusion_front, lake_at_rest,
                    relative_l2, ritter_dry_bed)
from .attribution import (FACTORS, LEGACY_TOTAL, LegSpec, anova_main_effects,
                          disagreement, evaluate_config, factorial_ledger,
                          legacy_replay, make_legs, order_table, shapley)
from .dg import SolverConfig, SolverResult, dg_solve_scalar, dg_solve_swe
from .fvm import fvm_solve_scalar, fvm_solve_swe
from .lbm import lbm_linear_wave, lbm_poiseuille
from .orders import expected_band, in_band, observed_order, order_study
from .sph import sph_poiseuille, sph_shallow_wave

__all__ = [
    "ScalarCase", "SWECase", "LinearWaveCase", "advection_sine",
    "diffusion_front", "lake_at_rest", "ritter_dry_bed", "cell_averages",
    "relative_l2", "SolverConfig", "SolverResult", "dg_solve_scalar",
    "dg_solve_swe", "fvm_solve_scalar", "fvm_solve_swe",
    "lbm_poiseuille", "lbm_linear_wave", "sph_poiseuille", "sph_shallow_wave",
    "FACTORS", "LEGACY_TOTAL", "LegSpec", "anova_main_effects", "disagreement",
    "evaluate_config", "factorial_ledger", "legacy_replay", "make_legs",
    "order_table", "shapley", "expected_band", "in_band", "observed_order",
    "order_study",
]