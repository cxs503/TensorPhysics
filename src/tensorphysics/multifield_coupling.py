"""Multi-field coupling (LBM↔FEM↔DEM) with energy conservation ledger.

This module implements partitioned coupling between three physics domains:
- LBM (Lattice Boltzmann Method): fluid domain
- FEM (Finite Element Method): structural domain  
- DEM (Discrete Element Method): granular/particle domain

Key features:
- Runtime energy ledger tracking energy fluxes across coupling interfaces
- Sub-cycling with interpolation order control (linear, quadratic, cubic)
- Conservation residual monitoring with configurable tolerance
- Energy flux conservation error tracking

References:
- Partitioned coupling strategies for multiphysics, Felippa & Park (2004)
- Energy conservation in partitioned coupling, Küttler & Wall (2008)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
import math


class CouplingDomain(str, Enum):
    """Physics domains in the multi-field coupling."""
    LBM = "lbm"
    FEM = "fem"
    DEM = "dem"


class InterpolationOrder(int, Enum):
    """Sub-cycling interpolation order."""
    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3


@dataclass
class EnergyFlux:
    """Energy flux across a coupling interface."""
    from_domain: CouplingDomain
    to_domain: CouplingDomain
    value: float  # Watts (J/s)
    timestamp: float  # seconds
    interface_id: str = ""

    def __post_init__(self) -> None:
        if self.from_domain == self.to_domain:
            raise ValueError("Energy flux must be between different domains")
        if not math.isfinite(self.value):
            raise ValueError("Energy flux must be finite")


@dataclass
class EnergyLedger:
    """Tracks energy conservation across coupling interfaces.
    
    The ledger maintains:
    - Total energy injected into the system
    - Total energy extracted from the system
    - Interface fluxes (with sign convention: positive = into receiving domain)
    - Conservation residual = |E_in - E_out - ΔE_stored|
    """
    total_energy_in: float = 0.0
    total_energy_out: float = 0.0
    interface_fluxes: list[EnergyFlux] = field(default_factory=list)
    domain_stored_energy: dict[str, float] = field(default_factory=dict)
    initial_energy: float = 0.0
    
    def record_injection(self, domain: CouplingDomain, energy: float) -> None:
        """Record energy injected into a domain (e.g., source term)."""
        self.total_energy_in += energy
        key = domain.value
        self.domain_stored_energy[key] = self.domain_stored_energy.get(key, 0.0) + energy
    
    def record_extraction(self, domain: CouplingDomain, energy: float) -> None:
        """Record energy extracted from a domain (e.g., dissipation)."""
        self.total_energy_out += energy
        key = domain.value
        self.domain_stored_energy[key] = self.domain_stored_energy.get(key, 0.0) - energy
    
    def record_interface_flux(self, flux: EnergyFlux) -> None:
        """Record energy transfer across an interface."""
        self.interface_fluxes.append(flux)
        # Update stored energy: source loses, target gains
        src_key = flux.from_domain.value
        tgt_key = flux.to_domain.value
        self.domain_stored_energy[src_key] = self.domain_stored_energy.get(src_key, 0.0) - flux.value
        self.domain_stored_energy[tgt_key] = self.domain_stored_energy.get(tgt_key, 0.0) + flux.value
    
    def set_initial_energy(self, energy: float, domain_distribution: dict[str, float] | None = None) -> None:
        """Set the initial total energy in the system.
        
        Args:
            energy: Total initial energy
            domain_distribution: Optional dict mapping domain names to initial energies.
                                If None, energy is not distributed to domains.
        """
        self.initial_energy = energy
        if domain_distribution is not None:
            self.domain_stored_energy = dict(domain_distribution)
        else:
            for domain in CouplingDomain:
                if domain.value not in self.domain_stored_energy:
                    self.domain_stored_energy[domain.value] = 0.0
    
    def conservation_residual(self) -> float:
        """Compute energy conservation residual.
        
        residual = |E_initial + E_in - E_out - ΣE_stored|
        
        For perfect conservation, this should be zero.
        """
        current_stored = sum(self.domain_stored_energy.values())
        expected = self.initial_energy + self.total_energy_in - self.total_energy_out
        return abs(expected - current_stored)
    
    def conservation_error_fraction(self) -> float:
        """Compute conservation error as fraction of total energy throughput."""
        throughput = max(abs(self.total_energy_in) + abs(self.total_energy_out), 
                        abs(self.initial_energy), 1e-30)
        return self.conservation_residual() / throughput
    
    def interface_balance_error(self) -> float:
        """Check that interface fluxes balance (action-reaction).
        
        For each interface pair, the flux from A→B should equal B→A in magnitude.
        Returns the maximum imbalance.
        """
        # Group fluxes by interface
        interface_sums: dict[str, float] = {}
        for flux in self.interface_fluxes:
            key = f"{flux.from_domain.value}->{flux.to_domain.value}"
            interface_sums[key] = interface_sums.get(key, 0.0) + flux.value
        
        # Check that for each A→B there's a corresponding B→A with opposite sign
        max_imbalance = 0.0
        checked = set()
        for flux in self.interface_fluxes:
            fwd_key = f"{flux.from_domain.value}->{flux.to_domain.value}"
            rev_key = f"{flux.to_domain.value}->{flux.from_domain.value}"
            if fwd_key in checked:
                continue
            fwd_sum = interface_sums.get(fwd_key, 0.0)
            rev_sum = interface_sums.get(rev_key, 0.0)
            imbalance = abs(fwd_sum + rev_sum)  # Should cancel if Newton's 3rd holds
            max_imbalance = max(max_imbalance, imbalance)
            checked.add(fwd_key)
            checked.add(rev_key)
        
        return max_imbalance


class DomainSolver(Protocol):
    """Protocol for a domain solver participating in multi-field coupling."""
    
    @property
    def domain(self) -> CouplingDomain:
        """The physics domain this solver represents."""
        ...
    
    @property
    def name(self) -> str:
        """Solver identifier."""
        ...
    
    def get_energy(self) -> float:
        """Return current total energy in this domain."""
        ...
    
    def advance(self, dt: float, interface_forces: dict[str, float]) -> None:
        """Advance the solver by dt with given interface forces/fluxes."""
        ...
    
    def get_interface_flux(self, target_domain: CouplingDomain) -> float:
        """Return the energy flux this domain sends to target_domain."""
        ...


@dataclass
class SubCyclingConfig:
    """Configuration for sub-cycling between domains."""
    ratio: int = 1  # Sub-steps per main step (e.g., 2, 4, 8)
    interpolation_order: InterpolationOrder = InterpolationOrder.LINEAR
    
    def __post_init__(self) -> None:
        if self.ratio < 1:
            raise ValueError("Sub-cycling ratio must be >= 1")
        if self.ratio > 64:
            raise ValueError("Sub-cycling ratio too large (max 64)")


def interpolate_temporal(values: list[float], times: list[float], t_query: float,
                         order: InterpolationOrder) -> float:
    """Temporal interpolation for sub-cycling.
    
    Args:
        values: Sampled values at given times
        times: Sample times (must be sorted ascending)
        t_query: Query time
        order: Interpolation order (1=linear, 2=quadratic, 3=cubic)
    
    Returns:
        Interpolated value at t_query
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 samples for interpolation")
    if len(values) != len(times):
        raise ValueError("values and times must have same length")
    
    # Clamp to range (no extrapolation)
    if t_query <= times[0]:
        return values[0]
    if t_query >= times[-1]:
        return values[-1]
    
    if order == InterpolationOrder.LINEAR:
        # Find bracketing interval
        for i in range(len(times) - 1):
            if times[i] <= t_query <= times[i + 1]:
                alpha = (t_query - times[i]) / (times[i + 1] - times[i])
                return values[i] * (1 - alpha) + values[i + 1] * alpha
        return values[-1]
    
    elif order == InterpolationOrder.QUADRATIC:
        # Lagrange quadratic through 3 nearest points
        if len(values) < 3:
            # Fall back to linear
            return interpolate_temporal(values, times, t_query, InterpolationOrder.LINEAR)
        
        # Find 3 nearest points
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t_query))
        start = max(0, min(idx - 1, len(times) - 3))
        pts = list(range(start, start + 3))
        
        result = 0.0
        for i, pi in enumerate(pts):
            basis = 1.0
            for j, pj in enumerate(pts):
                if i != j:
                    basis *= (t_query - times[pj]) / (times[pi] - times[pj])
            result += values[pi] * basis
        return result
    
    elif order == InterpolationOrder.CUBIC:
        # Lagrange cubic through 4 nearest points
        if len(values) < 4:
            return interpolate_temporal(values, times, t_query, InterpolationOrder.QUADRATIC)
        
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t_query))
        start = max(0, min(idx - 1, len(times) - 4))
        pts = list(range(start, start + 4))
        
        result = 0.0
        for i, pi in enumerate(pts):
            basis = 1.0
            for j, pj in enumerate(pts):
                if i != j:
                    basis *= (t_query - times[pj]) / (times[pi] - times[pj])
            result += values[pi] * basis
        return result
    
    raise ValueError(f"Unknown interpolation order: {order}")


@dataclass
class MultiFieldCouplingResult:
    """Result of a multi-field coupling simulation."""
    time_history: list[float]
    energy_history: dict[str, list[float]]  # domain -> energy at each time
    interface_flux_history: list[list[EnergyFlux]]  # fluxes at each time step
    conservation_residuals: list[float]
    final_ledger: EnergyLedger
    sub_cycle_ratios_tested: list[int] = field(default_factory=list)
    interpolation_errors: dict[int, float] = field(default_factory=dict)


class MultiFieldCouplingSimulator:
    """Partitioned multi-field coupling simulator with energy conservation.
    
    Implements a staggered (partitioned) coupling scheme where each domain
    solver advances independently and exchanges data at coupling interfaces.
    
    The simulator tracks:
    - Energy in each domain
    - Energy fluxes across interfaces
    - Conservation residuals
    - Sub-cycling interpolation errors
    """
    
    def __init__(self, solvers: list[DomainSolver], 
                 sub_cycling: SubCyclingConfig | None = None,
                 conservation_tolerance: float = 0.01) -> None:
        """
        Args:
            solvers: List of domain solvers (LBM, FEM, DEM)
            sub_cycling: Sub-cycling configuration
            conservation_tolerance: Maximum allowed conservation error fraction
        """
        if len(solvers) < 2:
            raise ValueError("Need at least 2 solvers for coupling")
        
        self.solvers = {s.domain: s for s in solvers}
        self.sub_cycling = sub_cycling or SubCyclingConfig()
        self.conservation_tolerance = conservation_tolerance
        self.ledger = EnergyLedger()
        
        # Coupling interface definitions (bidirectional)
        self.interfaces: list[tuple[CouplingDomain, CouplingDomain]] = []
        self._setup_default_interfaces()
    
    def _setup_default_interfaces(self) -> None:
        """Set up default coupling interfaces between domains."""
        domains = list(self.solvers.keys())
        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                self.interfaces.append((d1, d2))
                self.interfaces.append((d2, d1))
    
    def run(self, dt: float, total_time: float) -> MultiFieldCouplingResult:
        """Run the coupled simulation.
        
        Args:
            dt: Main time step
            total_time: Total simulation time
        
        Returns:
            MultiFieldCouplingResult with energy history and conservation data
        """
        n_steps = int(total_time / dt)
        time_history = [0.0]
        energy_history: dict[str, list[float]] = {d.value: [] for d in CouplingDomain}
        flux_history: list[list[EnergyFlux]] = []
        residuals: list[float] = []
        
        # Record initial state
        initial_distribution = {}
        for domain, solver in self.solvers.items():
            e = solver.get_energy()
            energy_history[domain.value].append(e)
            initial_distribution[domain.value] = e
        
        # Set initial energy in ledger with proper distribution
        initial_total = sum(initial_distribution.values())
        self.ledger.set_initial_energy(initial_total, initial_distribution)
        
        current_time = 0.0
        sub_dt = dt / self.sub_cycling.ratio
        
        for step in range(n_steps):
            step_fluxes: list[EnergyFlux] = []
            
            # Sub-cycling loop
            for sub_step in range(self.sub_cycling.ratio):
                sub_time = current_time + sub_step * sub_dt
                
                # Each solver advances with interface data
                for domain, solver in self.solvers.items():
                    # Collect interface fluxes for this domain
                    interface_forces: dict[str, float] = {}
                    for (src, tgt) in self.interfaces:
                        if tgt == domain and src != domain:
                            src_solver = self.solvers[src]
                            flux_value = src_solver.get_interface_flux(domain)
                            interface_forces[src.value] = flux_value
                    
                    solver.advance(sub_dt, interface_forces)
                
                # Record interface fluxes at end of sub-step
                sub_time_end = sub_time + sub_dt
                for (src, tgt) in self.interfaces:
                    if src < tgt:  # Avoid double counting
                        flux_val = self.solvers[src].get_interface_flux(tgt)
                        flux = EnergyFlux(
                            from_domain=src,
                            to_domain=tgt,
                            value=flux_val * sub_dt,
                            timestamp=sub_time_end,
                            interface_id=f"{src.value}-{tgt.value}"
                        )
                        step_fluxes.append(flux)
                        self.ledger.record_interface_flux(flux)
            
            current_time += dt
            time_history.append(current_time)
            
            # Record energies
            for domain, solver in self.solvers.items():
                energy_history[domain.value].append(solver.get_energy())
            
            flux_history.append(step_fluxes)
            
            # Compute conservation residual by comparing actual vs expected
            # Expected = initial + net interface transfers (which should be zero for closed system)
            actual_total = sum(solver.get_energy() for solver in self.solvers.values())
            # For a closed system with only interface transfers, total should be conserved
            # minus any dissipation in the solvers
            residual = abs(actual_total - initial_total)
            residuals.append(residual)
        
        return MultiFieldCouplingResult(
            time_history=time_history,
            energy_history=energy_history,
            interface_flux_history=flux_history,
            conservation_residuals=residuals,
            final_ledger=self.ledger,
        )
    
    def scan_sub_cycle_ratios(self, dt: float, total_time: float,
                              ratios: list[int] | None = None,
                              reference_solution: Callable[[float], float] | None = None
                              ) -> dict[int, float]:
        """Scan sub-cycling ratios and measure interpolation error.
        
        Args:
            dt: Main time step
            total_time: Total simulation time
            ratios: List of sub-cycling ratios to test (default: [1, 2, 4, 8])
            reference_solution: Optional reference solution for error computation
        
        Returns:
            Dictionary mapping ratio to interpolation error
        """
        if ratios is None:
            ratios = [1, 2, 4, 8]
        
        errors: dict[int, float] = {}
        
        for ratio in ratios:
            self.sub_cycling = SubCyclingConfig(ratio=ratio)
            self.ledger = EnergyLedger()
            
            result = self.run(dt, total_time)
            
            # Compute error as max conservation residual
            max_residual = max(result.conservation_residuals) if result.conservation_residuals else 0.0
            
            if reference_solution is not None:
                # Compare against reference at final time
                ref = reference_solution(total_time)
                actual = sum(result.energy_history[d][-1] for d in CouplingDomain if d.value in result.energy_history)
                max_residual = abs(actual - ref)
            
            errors[ratio] = max_residual
        
        return errors


# Concrete solver implementations for testing

@dataclass
class SimpleLBMSolver:
    """Simplified LBM solver for coupling tests."""
    name: str = "lbm"
    energy: float = 100.0  # Initial kinetic energy (J)
    coupling_coefficient: float = 0.1  # Energy exchange rate
    _dissipation_rate: float = 0.01
    
    @property
    def domain(self) -> CouplingDomain:
        return CouplingDomain.LBM
    
    def get_energy(self) -> float:
        return self.energy
    
    def advance(self, dt: float, interface_forces: dict[str, float]) -> None:
        # Dissipation
        self.energy -= self._dissipation_rate * dt * self.energy
        # Receive from interfaces
        for src, flux in interface_forces.items():
            self.energy += flux
    
    def get_interface_flux(self, target_domain: CouplingDomain) -> float:
        # Transfer energy proportional to coupling coefficient
        return self.coupling_coefficient * self.energy * 0.01


@dataclass
class SimpleFEMSolver:
    """Simplified FEM solver for coupling tests."""
    name: str = "fem"
    energy: float = 50.0  # Initial strain energy (J)
    coupling_coefficient: float = 0.05
    _damping_rate: float = 0.005
    
    @property
    def domain(self) -> CouplingDomain:
        return CouplingDomain.FEM
    
    def get_energy(self) -> float:
        return self.energy
    
    def advance(self, dt: float, interface_forces: dict[str, float]) -> None:
        # Damping
        self.energy -= self._damping_rate * dt * self.energy
        # Receive from interfaces
        for src, flux in interface_forces.items():
            self.energy += flux
    
    def get_interface_flux(self, target_domain: CouplingDomain) -> float:
        return self.coupling_coefficient * self.energy * 0.005


@dataclass
class SimpleDEMSolver:
    """Simplified DEM solver for coupling tests."""
    name: str = "dem"
    energy: float = 30.0  # Initial kinetic energy of particles (J)
    coupling_coefficient: float = 0.02
    _friction_rate: float = 0.008
    
    @property
    def domain(self) -> CouplingDomain:
        return CouplingDomain.DEM
    
    def get_energy(self) -> float:
        return self.energy
    
    def advance(self, dt: float, interface_forces: dict[str, float]) -> None:
        # Friction dissipation
        self.energy -= self._friction_rate * dt * self.energy
        # Receive from interfaces
        for src, flux in interface_forces.items():
            self.energy += flux
    
    def get_interface_flux(self, target_domain: CouplingDomain) -> float:
        return self.coupling_coefficient * self.energy * 0.003
