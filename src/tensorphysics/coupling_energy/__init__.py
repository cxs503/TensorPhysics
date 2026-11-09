"""Runtime energy/momentum conservation contract for partitioned multi-solver
coupling (campaign node ``P40W3C-N09``).

Public API
----------
Ledgers:
    :class:`~tensorphysics.coupling_energy.ledger.EnergyLedger`
        side energies minus external input plus dissipation; ``residual()``,
        ``max_residual()``, ``drift_slope()``, ``is_divergent()``.
    :class:`~tensorphysics.coupling_energy.ledger.MomentumLedger`
        impulse bookkeeping for force-coupled solvers.
Temporal exchange:
    :class:`~tensorphysics.coupling_energy.exchange.TemporalScheme`
        ``independent`` (each side evaluates its own interface), ``zoh``,
        ``linear``; :func:`...exchange.quadrature_error`,
        :func:`...exchange.reconstruction_order_scan`.
Solvers (minimum executables, no kernels duplicated from TensorLBM/FEM/DEM):
    :class:`...solvers.ThermalLBM2D`, :class:`...solvers.Q4ThermalFEM`,
    :class:`...solvers.Q4PlaneStressFEM`, :class:`...solvers.DEM2D`.
Scenarios (the three couplings this node verifies):
    :func:`~tensorphysics.coupling_energy.scenarios.run_thermal_coupling`
    :func:`...scenarios.run_mechanical_coupling`
    :func:`...scenarios.run_three_field`
    :func:`...scenarios.subcycling_scan`, :func:`...scenarios.solver_verification`.
Acceptance:
    :func:`~tensorphysics.coupling_energy.acceptance.run_acceptance`
        returns every criterion number quoted in the report.
"""
from .ledger import (CouplingLedgerError, EnergyLedger, EnergySample,
                     MomentumLedger, OrderEstimate, observed_order)
from .exchange import (MultiRateSchedule, ExchangeSample, SCHEME_ORDER,
                       TemporalScheme, quadrature_error,
                       reconstruction_order_scan, reconstruct_interval)
from .solvers import (ContactLaw, DEM2D, Q4Mesh, Q4PlaneStressFEM,
                      Q4ThermalFEM, ThermalLBM2D)
from .scenarios import (MechanicalConfig, ThermalConfig, ThreeFieldConfig,
                        gather_nodes_to_columns, interval_exchange, order_fit,
                        run_mechanical_coupling, run_thermal_coupling,
                        run_three_field, scatter_columns_to_nodes,
                        solver_verification, subcycling_scan)
from .acceptance import (GATES, run_acceptance)

__all__ = [
    "CouplingLedgerError", "EnergyLedger", "EnergySample", "MomentumLedger",
    "OrderEstimate", "observed_order", "MultiRateSchedule", "ExchangeSample",
    "SCHEME_ORDER", "TemporalScheme", "quadrature_error",
    "reconstruction_order_scan", "reconstruct_interval", "ContactLaw", "DEM2D",
    "Q4Mesh", "Q4PlaneStressFEM", "Q4ThermalFEM", "ThermalLBM2D",
    "MechanicalConfig", "ThermalConfig", "ThreeFieldConfig",
    "gather_nodes_to_columns", "interval_exchange", "order_fit",
    "run_mechanical_coupling", "run_thermal_coupling", "run_three_field",
    "scatter_columns_to_nodes", "solver_verification", "subcycling_scan",
    "GATES", "run_acceptance",
]
