"""Scheduling-pipeline integration: gate the solver registry before it is built.

The scheduler is solver-agnostic (:class:`tensorphysics.scheduler.Simulation`
only knows the ``Solver`` protocol).  This module inserts the evidence gate in
front of it:

* ``factories`` maps ``solver_id -> () -> Solver`` (lazy: a refused solver is
  never even instantiated);
* :func:`build_simulation` asks the router which solver to run, instantiates
  *only* that one, verifies the object really satisfies the protocol under the
  routed name, and registers it;
* when the gate refuses everything the call raises
  :class:`~.gate.RoutingRejected` -- the simulation is never silently built on a
  placeholder solver.

The returned :class:`RoutedSimulation` carries the routing decision so callers
can persist "which evidence level produced this run" next to the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from ..config import SimulationConfig
from ..scheduler import Simulation
from ..protocols import Solver
from .evidence import AdmissibilityPolicy
from .ledger import CapabilityLedger
from .router import CapabilityAwareRouter, RoutingDecision

__all__ = ["RoutedSimulation", "RoutingIntegrationError", "build_simulation", "route_solver"]

SOLVER_METHODS = ("initialize", "advance", "finalize", "snapshot")


class RoutingIntegrationError(RuntimeError):
    """A routed solver does not match the protocol contract of its ledger row."""


@dataclass(frozen=True)
class RoutedSimulation:
    """A scheduler instance plus the decision that produced its solver set."""

    simulation: Simulation
    decision: RoutingDecision

    @property
    def solver(self) -> Solver:
        return self.simulation.solvers[0]

    def manifest(self) -> dict:
        return {
            "capability_id": self.decision.capability_id,
            "solver_id": self.decision.selected,
            "evidence_level": self.decision.level.value if self.decision.level else None,
            "policy_tier": self.decision.tier,
            "decision": self.decision.as_dict(),
        }


def route_solver(capability_id: str,
                 factories: Mapping[str, Callable[[], Solver]],
                 ledger: "CapabilityLedger | None" = None,
                 router: "CapabilityAwareRouter | None" = None,
                 policy: "AdmissibilityPolicy | None" = None,
                 preferred: "str | None" = None,
                 unavailable: Iterable[str] = ()) -> tuple[Solver, RoutingDecision]:
    """Route + instantiate a single solver; raise rather than degrade silently."""
    if router is None:
        if ledger is None:
            raise TypeError("pass either router= or ledger=")
        router = CapabilityAwareRouter(ledger, policy)
    if not isinstance(factories, Mapping) or not factories:
        raise TypeError("factories must be a non-empty mapping solver_id -> factory")
    graded = {r.solver_id for r in router.ledger.candidates(capability_id)}
    missing = set(factories) - graded
    if missing:
        raise RoutingIntegrationError(
            f"factories for {sorted(missing)} have no ledger row under "
            f"capability {capability_id!r} (fail-closed: ungraded solver)"
        )
    decision = router.route(capability_id, preferred=preferred, unavailable=unavailable)
    factory = factories[decision.selected]
    solver = factory()
    for method in SOLVER_METHODS:
        if not callable(getattr(solver, method, None)):
            raise RoutingIntegrationError(
                f"solver {decision.selected!r} lacks callable {method}()"
            )
    if getattr(solver, "name", None) != decision.selected:
        raise RoutingIntegrationError(
            f"solver name {getattr(solver, 'name', None)!r} does not match the routed "
            f"id {decision.selected!r}; the decision record would be untraceable"
        )
    return solver, decision


def build_simulation(config: SimulationConfig,
                     capability_id: str,
                     factories: Mapping[str, Callable[[], Solver]],
                     ledger: "CapabilityLedger | None" = None,
                     router: "CapabilityAwareRouter | None" = None,
                     policy: "AdmissibilityPolicy | None" = None,
                     preferred: "str | None" = None,
                     unavailable: Iterable[str] = (),
                     extra_solvers: Iterable[Solver] = ()) -> RoutedSimulation:
    """Build a scheduler whose primary solver was chosen by the evidence gate."""
    solver, decision = route_solver(capability_id, factories, ledger=ledger, router=router,
                                    policy=policy, preferred=preferred,
                                    unavailable=unavailable)
    simulation = Simulation(config, [solver, *extra_solvers])
    return RoutedSimulation(simulation=simulation, decision=decision)