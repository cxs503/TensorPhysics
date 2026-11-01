"""SEA energy balance equation solver.

The fundamental SEA equation for N subsystems at a given frequency:

    omega * eta_i * n_i * E_i + omega * sum_j(eta_ij * n_i * E_i - eta_ji * n_j * E_j) = Pin_i

where:
    omega = 2*pi*f: angular frequency
    eta_i: internal loss factor of subsystem i
    n_i: modal density of subsystem i
    E_i: modal energy of subsystem i
    eta_ij: coupling loss factor from i to j
    Pin_i: input power to subsystem i

In matrix form: omega * [C] * {E} = {Pin}

where C_ii = eta_i * n_i + sum_j(eta_ij * n_i)
      C_ij = -eta_ji * n_j  (for i ≠ j)

Using reciprocity: eta_ji * n_j = eta_ij * n_i

Solution: {E} = (1/omega) * [C]^{-1} * {Pin}

From modal energy, response quantities are derived:
    - Mean-square velocity: <v²> = omega² * E_i / M_i
    - Sound power: W_rad = omega * eta_rad * E_i
    - Sound pressure: p² = rho*c² * E_cavity / V
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .subsystems import PlateSubsystem, BeamSubsystem, CavitySubsystem, Subsystem


@dataclass
class SEASubsystem:
    """SEA subsystem with computed properties."""
    subsystem: Subsystem
    modal_density: float = 0.0
    internal_loss_factor: float = 0.0
    coupling_loss_factors: dict[int, float] = field(default_factory=dict)
    input_power: float = 0.0
    modal_energy: float = 0.0
    vibration_velocity_rms: float = 0.0
    sound_power: float = 0.0


@dataclass
class SEASystem:
    """Complete SEA system definition.

    Attributes:
        subsystems: list of SEA subsystems.
        coupling_matrix: dict of (i, j) -> eta_ij coupling loss factors.
        frequency_bands: list of center frequencies (Hz) for analysis.
    """
    subsystems: list[SEASubsystem] = field(default_factory=list)
    coupling_matrix: dict[tuple[int, int], float] = field(default_factory=dict)
    frequency_bands: list[float] = field(default_factory=list)

    def add_subsystem(self, sea_sub: SEASubsystem) -> None:
        self.subsystems.append(sea_sub)

    def set_coupling(self, i: int, j: int, eta_ij: float) -> None:
        """Set coupling loss factor from subsystem i to j."""
        if i == j:
            raise ValueError("Cannot set self-coupling")
        self.coupling_matrix[(i, j)] = eta_ij

    def solve_at_frequency(self, frequency: float) -> list[float]:
        """Solve SEA equations at a single frequency.

        Returns:
            List of modal energies for each subsystem.
        """
        if frequency <= 0:
            raise ValueError("frequency must be positive")

        n = len(self.subsystems)
        if n == 0:
            return []

        omega = 2.0 * math.pi * frequency

        # Build the SEA matrix [C] and input power vector {Pin}
        C = [[0.0] * n for _ in range(n)]
        Pin = [0.0] * n

        for i in range(n):
            sub = self.subsystems[i]
            ni = sub.modal_density
            eta_i = sub.internal_loss_factor

            # Diagonal: eta_i * n_i + sum_j(eta_ij * n_i)
            C[i][i] = eta_i * ni
            for j in range(n):
                if i != j:
                    eta_ij = self.coupling_matrix.get((i, j), 0.0)
                    C[i][i] += eta_ij * ni

            # Off-diagonal: -eta_ji * n_j = -eta_ij * n_i (reciprocity)
            for j in range(n):
                if i != j:
                    eta_ij = self.coupling_matrix.get((i, j), 0.0)
                    C[i][j] = -eta_ij * ni

            # Input power
            Pin[i] = sub.input_power

        # Solve [C]{E} = {Pin}/omega
        # {E} = (1/omega) * [C]^{-1} * {Pin}
        energies = _solve_linear_system(C, Pin, omega)

        # Store results
        for i in range(n):
            self.subsystems[i].modal_energy = energies[i]

        return energies

    def solve_all_frequencies(self) -> dict[float, list[float]]:
        """Solve SEA equations at all frequency bands.

        Returns:
            Dict mapping frequency to list of modal energies.
        """
        results = {}
        for f in self.frequency_bands:
            energies = self.solve_at_frequency(f)
            results[f] = energies
        return results


def _solve_linear_system(
    C: list[list[float]],
    Pin: list[float],
    omega: float,
) -> list[float]:
    """Solve [C]{E} = {Pin}/omega using Gaussian elimination.

    Args:
        C: N×N coefficient matrix.
        Pin: input power vector.
        omega: angular frequency.

    Returns:
        Solution vector {E}.
    """
    n = len(C)
    if n == 0:
        return []

    # Augmented matrix [C | Pin/omega]
    aug = [row[:] + [Pin[i] / omega] for i, row in enumerate(C)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        # Swap rows
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-15:
            raise ValueError("SEA matrix is singular or near-singular")

        # Eliminate below
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x


def solve_sea(
    subsystems: list[SEASubsystem],
    coupling_matrix: dict[tuple[int, int], float],
    frequency_bands: list[float],
) -> dict[float, list[float]]:
    """Convenience function to solve a complete SEA problem.

    Args:
        subsystems: list of SEA subsystems with properties set.
        coupling_matrix: dict of (i,j) -> eta_ij.
        frequency_bands: center frequencies for analysis.

    Returns:
        Dict mapping frequency to modal energies.
    """
    system = SEASystem(
        subsystems=subsystems,
        coupling_matrix=coupling_matrix,
        frequency_bands=frequency_bands,
    )
    return system.solve_all_frequencies()
