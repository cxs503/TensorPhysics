"""Time-domain spectral plate / potential-flow solver for an ice cover.

Model
-----
Kirchhoff-Love ice plate of areal mass ``rho_i h``, flexural rigidity ``D`` and
damping ``c`` resting on an inviscid water layer of depth ``H`` (rigid bottom).
The fluid is eliminated with the wavenumber-dependent added-mass operator of
:mod:`tensorphysics.ice.dispersion` (potential flow, Laplace + ice-water
kinematic condition + impermeable bottom).  In a 2-D Fourier mode the coupled
system is a single damped oscillator

    m(k) w'' + c w' + K(k) w = p_hat(k, t),
    m(k) = rho_i h + m_a(k),   K(k) = D k^4 + rho_w g,

where ``w`` is positive downwards and ``p`` is the downward pressure.  Modes are
independent, so the semi-discrete system is integrated with the Newmark
average-acceleration scheme (``beta = 1/4``, ``gamma = 1/2``), which is
unconditionally stable and non-dissipative for linear systems.

Energy (per unit area, summed over modes by Parseval)
-----------------------------------------------------
    E_bend = 1/2 D k^4 |w_hat|^2,     E_buoy = 1/2 rho_w g |w_hat|^2,
    E_ice  = 1/2 rho_i h |v_hat|^2,   E_fluid = 1/2 m_a(k) |v_hat|^2,

all divided by the number of grid points.  The damping power is
``c |v_hat|^2``; the fluid kinetic energy uses the *same symmetric operator*
that appears in the momentum equation, which is why the semi-discrete system is
energy consistent by construction (see :meth:`energies`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dispersion import (
    GRAVITY,
    RHO_ICE,
    RHO_WATER,
    added_mass_per_area,
    plate_rigidity,
)


@dataclass(frozen=True)
class IceWaterParameters:
    """Physical parameters of the ice cover / water system (SI units)."""

    E: float = 9.0e9          # effective (strain-rate) Young modulus of ice [Pa]
    h: float = 1.0            # ice thickness [m]
    nu: float = 0.3           # Poisson ratio
    rho_ice: float = RHO_ICE
    rho_w: float = RHO_WATER
    g: float = GRAVITY
    depth: float | None = 50.0
    damping: float = 0.0      # distributed damping c [Pa s / m]

    @property
    def rigidity(self) -> float:
        return plate_rigidity(self.E, self.h, self.nu)

    @property
    def areal_mass(self) -> float:
        return self.rho_ice * self.h

    def as_dict(self) -> dict:
        return {
            "E_Pa": self.E,
            "h_m": self.h,
            "nu": self.nu,
            "rho_ice_kg_m3": self.rho_ice,
            "rho_w_kg_m3": self.rho_w,
            "g_m_s2": self.g,
            "depth_m": self.depth,
            "damping_Pa_s_m": self.damping,
            "rigidity_Nm": self.rigidity,
            "areal_mass_kg_m2": self.areal_mass,
        }


@dataclass
class CoupledPlateFluid:
    """Spectral (Fourier) plate-fluid system with a Newmark time integrator."""

    params: IceWaterParameters = field(default_factory=IceWaterParameters)
    nx: int = 128
    ny: int = 128
    lx: float = 384.0
    ly: float = 384.0
    kind: str = "spectral"   # 'spectral' (continuous k^4) or 'finite_difference'

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("grid too small")
        if self.lx <= 0 or self.ly <= 0:
            raise ValueError("domain must be positive")
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="xy")
        self.K2 = self.KX**2 + self.KY**2
        if self.kind == "spectral":
            self.k4 = self.K2**2
        elif self.kind == "finite_difference":
            lap_h = (
                2.0 * np.cos(self.KX * self.dx) + 2.0 * np.cos(self.KY * self.dy) - 4.0
            ) / self.dx**2
            self.k4 = lap_h**2
        else:
            raise ValueError(f"unknown kind {self.kind!r}")
        self.k_mag = np.sqrt(self.K2)
        p = self.params
        self.K = p.rigidity * self.k4 + p.rho_w * p.g       # [Pa/m]
        self.ma = added_mass_per_area(self.k_mag, rho_w=p.rho_w, depth=p.depth)
        self.ma = np.where(np.isfinite(self.ma), self.ma, 0.0)  # k=0 masked out
        self.mass = p.areal_mass + self.ma                   # [kg/m^2]
        self.c = np.full_like(self.K, float(p.damping))
        # k = 0 (uniform vertical translation) excluded: infinite added mass
        self.mask = np.ones_like(self.K, dtype=bool)
        self.mask[0, 0] = False
        self.K[~self.mask] = 1.0
        self.mass[~self.mask] = 1.0
        self.t = 0.0
        self.W = np.zeros((self.ny, self.nx), dtype=complex)
        self.V = np.zeros_like(self.W)
        self.A = np.zeros_like(self.W)
        self._cached_dt: float | None = None
        self._phase_cache: dict = {}

    # ------------------------------------------------------------- utilities
    @property
    def n_grid(self) -> int:
        return self.nx * self.ny

    def wavenumber_vector(self) -> np.ndarray:
        return self.k_mag

    def characteristic_length(self) -> float:
        from .dispersion import characteristic_length

        return characteristic_length(self.params.rigidity, self.params.rho_w, self.params.g)

    def set_deflection(self, w: np.ndarray, velocity: np.ndarray | None = None) -> None:
        """Initialise from physical fields (zero-mean enforced)."""
        w_hat = np.fft.fft2(np.asarray(w, dtype=float))
        w_hat[0, 0] = 0.0
        self.W = w_hat
        if velocity is None:
            self.V = np.zeros_like(self.W)
        else:
            self.V = np.fft.fft2(np.asarray(velocity, dtype=float))
            self.V[0, 0] = 0.0
        self.A = (-self.K * self.W - self.c * self.V) / self.mass
        self.A[~self.mask] = 0.0
        self.W[~self.mask] = 0.0
        self.V[~self.mask] = 0.0

    def patch_load_hat(
        self,
        total_force: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float | None = None,
    ) -> np.ndarray:
        """FFT of a Gaussian contact patch carrying ``total_force`` [N] at (x0, y0).

        The patch has standard deviations ``sigma_x``/``sigma_y`` [m]; its peak
        pressure is ``F / (2 pi sigma_x sigma_y)``.  Gaussian patches are used
        because they are smooth and spectrally compact, so the contact pressure
        is resolved without Gibbs contamination.

        The returned array is the *discrete* Fourier transform of the sampled
        pressure field, i.e. the continuous transform divided by the cell area
        ``dx dy``; this is what the modal equation ``m w'' + K w = p`` expects.
        """
        sigma_y = sigma_x if sigma_y is None else sigma_y
        phase = np.exp(-1j * (self.KX * x0 + self.KY * y0))
        shape = np.exp(-0.5 * (self.KX * sigma_x) ** 2 - 0.5 * (self.KY * sigma_y) ** 2)
        return total_force * phase * shape / (self.dx * self.dy)

    def patch_pressure_peak(self, total_force: float, sigma_x: float, sigma_y: float | None = None) -> float:
        sigma_y = sigma_x if sigma_y is None else sigma_y
        return total_force / (2.0 * np.pi * sigma_x * sigma_y)

    # ------------------------------------------------------------- time step
    def step(self, dt: float | None = None, force_hat_next: np.ndarray | None = None,
             beta: float = 0.25, gamma: float = 0.5) -> None:
        """One Newmark average-acceleration step with the prescribed pressure.

        ``dt`` may be omitted after :meth:`_prepare` (constant time step), which
        precomputes the constant Newmark coefficients.
        """
        if force_hat_next is None:
            raise ValueError("force_hat_next is required")
        if dt is None:
            if self._cached_dt is None:
                raise ValueError("dt must be supplied on the first call")
            dt = self._cached_dt
        elif self._cached_dt != dt:
            self._prepare(dt, beta, gamma)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        f = np.asarray(force_hat_next, dtype=complex)
        if f.shape != self.W.shape:
            raise ValueError("force_hat shape mismatch")
        f = f.copy()
        f[0, 0] = 0.0                      # k = 0 (uniform translation) excluded
        W, V, A = self.W, self.V, self.A
        tmp = W + self._dt * V + self._c2 * A
        # single evaluation of the new acceleration, then the two updates
        acc = (f - self.c * (V + self._c1 * A) - self.K * tmp) * self._inv_denom
        W_new = tmp + self._c3 * acc
        V_new = V + self._c1 * A + self._c4 * acc
        W_new[0, 0] = 0.0
        V_new[0, 0] = 0.0
        acc[0, 0] = 0.0
        self.A, self.W, self.V = acc, W_new, V_new
        self.t += dt

    def _prepare(self, dt: float, beta: float = 0.25, gamma: float = 0.5) -> None:
        """Cache the constant Newmark coefficients for a fixed time step."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._cached_dt = dt
        self._dt = dt
        self._c1 = (1.0 - gamma) * dt
        self._c2 = (1.0 - 2.0 * beta) * 0.5 * dt * dt
        self._c3 = beta * dt * dt
        self._c4 = gamma * dt
        self._inv_denom = 1.0 / (self.mass + gamma * dt * self.c + beta * dt * dt * self.K)

    def step_to(self, t_next: float, force_fn, **kwargs) -> None:
        """Advance to ``t_next`` calling ``force_fn(t) -> force_hat``."""
        dt = t_next - self.t
        if dt <= 0.0:
            raise ValueError("t_next must exceed the current time")
        self.step(dt, force_fn(t_next), **kwargs)

    # ------------------------------------------------------------- diagnostics
    def field(self, which: str = "w") -> np.ndarray:
        """Physical field: ``w`` deflection [m], ``v`` velocity, ``a`` acceleration."""
        arr = {"w": self.W, "v": self.V, "a": self.A}[which]
        return np.real(np.fft.ifft2(arr))

    def value_at(self, x0: float, y0: float, which: str = "w") -> float:
        """Value of a physical field at one point, without a full inverse FFT.

        Uses the pre-computed phase table for repeated queries at the same
        location (contact point probing), which is ~10x cheaper than ifft2.
        """
        key = (which, round(float(x0), 9), round(float(y0), 9))
        phase = self._phase_cache.get(key)
        if phase is None:
            phase = np.exp(1j * (self.KX * x0 + self.KY * y0))
            if len(self._phase_cache) < 32:
                self._phase_cache[key] = phase
        arr = {"w": self.W, "v": self.V, "a": self.A}[which]
        return float(np.real(np.sum(arr * phase)) / self.n_grid)

    def derivatives(self, which: str = "w") -> dict:
        """Spectral first and second derivatives of a physical field."""
        arr = {"w": self.W, "v": self.V, "a": self.A}[which]
        d = {}
        d["x"] = np.real(np.fft.ifft2(1j * self.KX * arr))
        d["y"] = np.real(np.fft.ifft2(1j * self.KY * arr))
        d["xx"] = np.real(np.fft.ifft2(-self.KX**2 * arr))
        d["yy"] = np.real(np.fft.ifft2(-self.KY**2 * arr))
        d["xy"] = np.real(np.fft.ifft2(-self.KX * self.KY * arr))
        return d

    def energies(self) -> dict:
        """Modal energy per unit area, multiplied by the domain area [J]."""
        area = self.lx * self.ly
        n = self.n_grid
        # Parseval: int f^2 dA = (area / n^2) * sum_k |f_hat_k|^2
        wgt = area / n**2
        w2 = np.abs(self.W) ** 2
        v2 = np.abs(self.V) ** 2
        w2[~self.mask] = 0.0
        v2[~self.mask] = 0.0
        bend = 0.5 * self.params.rigidity * np.sum(self.k4 * w2) * wgt
        buoy = 0.5 * self.params.rho_w * self.params.g * np.sum(w2) * wgt
        ice_ke = 0.5 * self.params.areal_mass * np.sum(v2) * wgt
        fluid_ke = 0.5 * np.sum(self.ma * v2) * wgt
        total = bend + buoy + ice_ke + fluid_ke
        return {
            "bending_J": float(bend),
            "buoyancy_J": float(buoy),
            "ice_kinetic_J": float(ice_ke),
            "fluid_kinetic_J": float(fluid_ke),
            "total_J": float(total),
        }

    def dissipation_power(self) -> float:
        """Instantaneous damping power ``int c v^2 dA`` [W]."""
        area = self.lx * self.ly
        n = self.n_grid
        v2 = np.abs(self.V) ** 2
        v2[~self.mask] = 0.0
        return float(self.params.damping * np.sum(v2) * area / n**2)

    def fluid_pressure(self) -> np.ndarray:
        """Underwater (radiation) pressure field just below the ice [Pa].

        The potential-flow reaction is ``p_f = -m_a a`` (positive downwards), so
        an upward push on the plate corresponds to ``p_f < 0``.
        """
        return np.real(np.fft.ifft2(-self.ma * self.A))

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        return np.meshgrid(x, y, indexing="xy")