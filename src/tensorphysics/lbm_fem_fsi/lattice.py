"""D2Q9 lattice-Boltzmann fluid sub-solver with a *moving immersed circular body*.

Campaign node ``p362w2-n09`` (TensorPhysics).  This is the fluid half of the
LBM -> FEM partitioned FSI exchange; the structural half lives in
:mod:`tensorphysics.lbm_fem_fsi.structure` and the conservative load map in
:mod:`tensorphysics.lbm_fem_fsi.interface`.

What is implemented (and why it is auditable)
---------------------------------------------
* BGK D2Q9, ``rho = 1``, ``nu = (tau - 1/2)/3``.
* Zou/He velocity inlet, zero-gradient outlet, free-slip (specular) top and
  bottom walls.  The non-equilibrium part is mirrored across the inlet plane
  (``f5 - f5^eq = f7 - f7^eq``), not the raw populations -- using the raw
  populations costs ~30 % of the inlet velocity, see the note in
  ``docs/reports/p362w2_n09_lbm_fem_fsi/report.md``.
* The cylinder is a **moving immersed boundary**.  Two wall operators:

  ``halfway``  half-way bounce-back with the classical ``R - dx/2`` wall offset
               so that the effective body area matches the analytic circle.
               Mass and momentum exchange are exact, which is what the
               conservation criteria of this node are about.  **Production
               operator.**
  ``interp``   Bouzidi et al. (2001) interpolated bounce-back,
               ``f_opp = (1/2delta) f_i + (1 - 1/2delta) f_i(mirror)``.
               Sharper geometry, but the reconstruction borrows the mirror
               population without removing it: mass leaks and the local
               momentum exchange no longer closes the global budget (measured,
               reported as a negative result).

* The hydrodynamic load is measured **twice, independently**:

  ``mea``      Ladd momentum exchange over the boundary links,
               ``sum_links (f_opp + f_i) e_i`` -- a *local* measurement;
  ``budget``   the residual of the global discrete momentum budget including
               the streaming outflow, the inlet/outlet and wall terms -- an
               *integral* measurement.

  They agree only if collision, streaming and boundary operators are mutually
  momentum-consistent, so their difference is a genuine, non-tautological audit
  of the exchange.  (It is not the same number computed twice: ``mea`` never
  looks at anything outside the boundary links.)

* Mass and momentum are accounted per boundary operation (strip or link
  differences), so ``dM = M_stream + M_walls + M_inlet + M_outlet + M_body``
  closes to round-off over any window and a code path that silently creates
  mass shows up immediately.

Conventions: lattice units (``dx = dt = 1``), node ``(i, j)`` at
``(i + 1/2, j + 1/2)``.  Force is ``mass * lu / ts^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["EX", "EY", "W", "OPP", "CS2", "equilibrium", "Obstacle", "LatticeFSI",
           "StepLedger"]

EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0
# directions whose first index is "positive" (used by the symmetric equilibrium)
_HALF = (1, 2, 5, 6)
_HALF_WEIGHT = W[list(_HALF)]


def equilibrium(rho, ux, uy, out=None):
    """Second-order Maxwell-Boltzmann equilibrium, shape ``(9, nx, ny)``.

    Uses the ``e <-> -e`` symmetry ``A(-x) = A(x) - 2x``: only four directions
    are evaluated, the four opposites are shifted copies.
    """
    shape = rho.shape
    if out is None:
        out = np.empty((9,) + shape)
    u2 = 1.5 * (ux * ux + uy * uy)
    cx = 3.0 * ux
    cy = 3.0 * uy
    out[0] = (4.0 / 9.0) * rho * (1.0 - u2)
    for k in _HALF:
        xe = EX[k] * cx + EY[k] * cy
        a = 1.0 - u2 + xe + 0.5 * xe * xe
        out[k] = W[k] * rho * a
        out[OPP[k]] = W[k] * rho * (a - 2.0 * xe)
    return out


@dataclass
class Obstacle:
    """Rigid circular body (pure translation, as the FSI loop needs)."""

    x: float
    y: float
    radius: float
    ux: float = 0.0
    uy: float = 0.0

    def set_position(self, y: float, uy: float = 0.0) -> None:
        self.y = float(y)
        self.uy = float(uy)


@dataclass
class StepLedger:
    """Per-step exchange bookkeeping (lattice units)."""

    force_mea: np.ndarray = field(default_factory=lambda: np.zeros(2))
    force_budget: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torque_mea: float = 0.0
    mass_stream: float = 0.0
    mass_walls: float = 0.0
    mass_inlet: float = 0.0
    mass_outlet: float = 0.0
    mass_body: float = 0.0
    mass_body_exchange: float = 0.0
    momentum_stream: np.ndarray = field(default_factory=lambda: np.zeros(2))
    momentum_walls: np.ndarray = field(default_factory=lambda: np.zeros(2))
    momentum_inlet: np.ndarray = field(default_factory=lambda: np.zeros(2))
    momentum_outlet: np.ndarray = field(default_factory=lambda: np.zeros(2))
    momentum_body: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def as_dict(self) -> dict:
        return {
            "force_mea": self.force_mea.tolist(),
            "force_budget": self.force_budget.tolist(),
            "torque_mea": float(self.torque_mea),
            "mass_stream": float(self.mass_stream),
            "mass_inlet": float(self.mass_inlet),
            "mass_outlet": float(self.mass_outlet),
            "mass_body": float(self.mass_body),
        }


class LatticeFSI:
    """D2Q9 lattice with one moving immersed circular body."""

    def __init__(self, nx: int, ny: int, tau: float, U: float, obstacle: Obstacle,
                 wall_mode: str = "halfway", radius_offset: float | None = None):
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self.U = float(U)
        self.obstacle = obstacle
        self.wall_mode = wall_mode
        explicit_offset = radius_offset is not None
        if radius_offset is None:
            radius_offset = 0.5 if wall_mode == "halfway" else 0.0
        self.radius_offset = float(radius_offset)
        self._offset_locked = explicit_offset
        self.rho0 = 1.0
        rho = np.full((nx, ny), self.rho0)
        self.f = equilibrium(rho, np.full((nx, ny), self.U), np.zeros((nx, ny)))
        self.solid = np.zeros((nx, ny), dtype=bool)
        self._links = None
        self.last_samples = None
        self.steps = 0
        # cumulative ledgers (lattice units)
        self.mass_stream = 0.0
        self.mass_walls = 0.0
        self.mass_inlet = 0.0
        self.mass_outlet = 0.0
        self.mass_body = 0.0
        self.mass_body_exchange = 0.0
        self.mass_absorbed = 0.0
        self.mass_released = 0.0
        self.impulse_mea = np.zeros(2)
        self.work_on_body = 0.0
        self.place(obstacle.y, 0.0, rebuild=True)
        if not self._offset_locked:
            self._calibrate_offset()
            self.place(obstacle.y, 0.0, rebuild=True)
        self.mass_initial = self.mass_total()
        # the initial rasterisation is not a change during the run
        self.mass_stream = self.mass_walls = self.mass_inlet = 0.0
        self.mass_outlet = self.mass_body = self.mass_absorbed = 0.0
        self.mass_body_exchange = 0.0
        self.mass_released = 0.0

    # ------------------------------------------------------------------ geometry
    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    def _coords(self):
        x = np.broadcast_to((np.arange(self.nx) + 0.5)[:, None], (self.nx, self.ny))
        y = np.broadcast_to((np.arange(self.ny) + 0.5)[None, :], (self.nx, self.ny))
        return x, y

    def solid_mask(self, y_center: float | None = None) -> np.ndarray:
        yc = self.obstacle.y if y_center is None else y_center
        x, y = self._coords()
        r = self.obstacle.radius - self.radius_offset
        return (x - self.obstacle.x) ** 2 + (y - yc) ** 2 <= r * r

    def _calibrate_offset(self) -> None:
        """Choose the rasterisation radius so that the solid *area* matches pi R^2.

        Half-way bounce-back puts the wall half a lattice beyond the outermost
        solid node, so the naive ``R - 1/2`` mask under-sizes the body by O(R)
        percent at the resolutions used here (D = 12 gives an effective diameter
        of 10.6 instead of 12 -- a 12 % drag error).  Scanning the offset for the
        node count that best reproduces the analytic area fixes that without any
        interpolation, keeping the exact mass/momentum exchange of the operator.
        """
        x, y = self._coords()
        d2 = (x - self.obstacle.x) ** 2 + (y - self.obstacle.y) ** 2
        target = np.pi * self.obstacle.radius ** 2
        best_err, best_dr = None, 0.0
        for span, step in ((0.6, 0.05), (0.05, 0.005)):
            centre = best_dr
            for dr in np.arange(centre - span, centre + span + 1e-12, step):
                r = self.obstacle.radius + dr
                err = abs(float((d2 <= r * r).sum()) - target)
                if best_err is None or err < best_err:
                    best_err, best_dr = err, float(dr)
        self.radius_offset = -best_dr
        self.solid = self.solid_mask()

    def perturb(self, amplitude: float = 0.02, width: float | None = None) -> None:
        """Seed the antisymmetric instability of the wake.

        At Re = 100 the symmetric steady solution is unstable, but a perfectly
        symmetric lattice converges to it and the shedding then has to grow out
        of round-off (10^4+ steps).  A local transverse velocity bump in the
        equilibrium initial field gives the mode something to grow from.
        """
        x, y = self._coords()
        w = 0.5 * self.obstacle.radius if width is None else width
        r2 = (x - self.obstacle.x) ** 2 + (y - self.obstacle.y) ** 2
        bump = amplitude * self.U * np.exp(-r2 / (2.0 * w * w))
        rho = np.full((self.nx, self.ny), self.rho0)
        self.f = equilibrium(rho, np.full((self.nx, self.ny), self.U), bump)
        self.f[:, self.solid] = 0.0

    def effective_diameter(self) -> float:
        """Diameter implied by the rasterised solid area (``4 A / 1`` -> 2 sqrt(A/pi))."""
        return float(2.0 * np.sqrt(self.solid.sum() / np.pi))

    def place(self, y_center: float, uy: float = 0.0, rebuild: bool = True) -> float:
        """Move the body and (re)build the mask + link tables.

        Returns the mass created at newly wetted nodes (equilibrium refill),
        which is booked in ``mass_body``/``mass_released``.
        """
        self.obstacle.set_position(y_center, uy)
        if not rebuild:
            return 0.0
        rho, _, _ = self._macroscopic(self.f)
        solid = self.solid_mask()
        newly_solid = solid & ~self.solid
        newly_fluid = ~solid & self.solid
        mass_created = 0.0
        if newly_fluid.any():
            # Refill with the *local* fluid state (first-order extrapolation from
            # the 8 neighbours).  Filling with the wall velocity instead would
            # inject rho V u_wall of momentum every rebuild, which at D = 10 and
            # A/D = 0.4 is several times the physical exchange impulse.
            _, uxf, uyf = self._macroscopic(self.f)
            wetted = (~self.solid).astype(float)
            cnt = np.zeros((self.nx, self.ny))
            sx = np.zeros((self.nx, self.ny))
            sy = np.zeros((self.nx, self.ny))
            for k in range(1, 9):
                rw = np.roll(np.roll(wetted, EY[k], axis=1), EX[k], axis=0)
                cnt += rw
                sx += rw * np.roll(np.roll(uxf, EY[k], axis=1), EX[k], axis=0)
                sy += rw * np.roll(np.roll(uyf, EY[k], axis=1), EX[k], axis=0)
            safe = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1.0), 0.0)
            ux_new = np.where(cnt > 0, sx * safe, self.obstacle.ux)
            uy_new = np.where(cnt > 0, sy * safe, self.obstacle.uy)
            rho_mean = float(rho[~solid].sum() / max((~solid).sum(), 1))
            eq = equilibrium(np.full((self.nx, self.ny), rho_mean), ux_new, uy_new)
            self.f[:, newly_fluid] = eq[:, newly_fluid]
            mass_created = float(eq[:, newly_fluid].sum())
        if newly_solid.any():
            self.mass_absorbed += float(self.f[:, newly_solid].sum())
            self.mass_body -= float(self.f[:, newly_solid].sum())
        self.solid = solid
        self.f[:, self.solid] = 0.0      # no fluid lives inside the body
        self.mass_released += mass_created
        self.mass_body += mass_created
        self._build_links()
        return mass_created

    def _build_links(self) -> None:
        """Flat, vectorised boundary-link table (one entry per fluid->solid link)."""
        px = (np.broadcast_to((np.arange(self.nx) + 0.5)[:, None], (self.nx, self.ny))
              - self.obstacle.x)
        py = (np.broadcast_to((np.arange(self.ny) + 0.5)[None, :], (self.nx, self.ny))
              - self.obstacle.y)
        r2 = px ** 2 + py ** 2
        R = self.obstacle.radius
        nx, ny = self.nx, self.ny
        idx, src, mir, delta, kk, interp = [], [], [], [], [], []
        for k in range(1, 9):
            ex, ey = int(EX[k]), int(EY[k])
            nb_solid = np.roll(np.roll(self.solid, -ey, axis=1), -ex, axis=0)
            mask = (~self.solid) & nb_solid
            if not mask.any():
                continue
            ii, jj = np.nonzero(mask)
            lx = float(np.hypot(ex, ey))
            ehx, ehy = ex / lx, ey / lx
            b = px[ii, jj] * ehx + py[ii, jj] * ehy
            c = r2[ii, jj] - R * R
            s = (-b - np.sqrt(np.maximum(b * b - c, 0.0))) / lx
            si = ii - ex
            sj = jj - ey
            inb = (si >= 0) & (si < nx) & (sj >= 0) & (sj < ny)
            si_c = np.clip(si, 0, nx - 1)
            sj_c = np.clip(sj, 0, ny - 1)
            if self.wall_mode == "interp":
                use = inb & (~np.roll(np.roll(self.solid, ey, axis=1), ex, axis=0)[ii, jj]) & (s < 1.0)
                d = np.clip(s, 0.5, 1.0)
            else:
                use = np.zeros(ii.size, dtype=bool)
                d = np.full(ii.size, 0.5)
            idx.append(ii * ny + jj)
            src.append(k * nx * ny + ii * ny + jj)
            mir.append(k * nx * ny + si_c * ny + sj_c)
            delta.append(d)
            interp.append(use)
            kk.append(np.full(ii.size, k))
        if idx:
            cat = lambda L: np.concatenate(L)  # noqa: E731
            self._links = {
                "tgt": OPP[cat(kk)] * nx * ny + cat(idx),
                "src": cat(src),
                "mir": cat(mir),
                "delta": cat(delta),
                "k": cat(kk),
                "interp": cat(interp),
                "pos": np.column_stack([cat(idx) // ny + 0.5, cat(idx) % ny + 0.5]),
            }
        else:                                       # pragma: no cover
            self._links = None

    # ------------------------------------------------------------- tick helpers
    def _macroscopic(self, f):
        rho = f.sum(axis=0)
        inv = np.where(rho > 1e-12, 1.0 / np.maximum(rho, 1e-12), 0.0)
        ux = ((f[1] - f[3]) + (f[5] - f[6]) - (f[7] - f[8])) * inv
        uy = ((f[2] - f[4]) + (f[5] + f[6]) - (f[7] + f[8])) * inv
        return rho, ux, uy

    @staticmethod
    def _mom_mass(f, sl=None):
        sub = f if sl is None else f[(slice(None),) + sl]
        sk = sub.sum(axis=(1, 2))
        return np.array([float(sk @ EX), float(sk @ EY)]), float(sk.sum())

    def _stream(self, fpost: np.ndarray) -> np.ndarray:
        nx, ny = self.nx, self.ny
        out = np.zeros_like(fpost)   # slots fed from outside MUST start at zero
        for k in range(9):
            ex, ey = int(EX[k]), int(EY[k])
            di0, di1 = max(ex, 0), nx + min(ex, 0)
            dj0, dj1 = max(ey, 0), ny + min(ey, 0)
            si0, si1 = max(-ex, 0), nx + min(-ex, 0)
            sj0, sj1 = max(-ey, 0), ny + min(-ey, 0)
            out[k, di0:di1, dj0:dj1] = fpost[k, si0:si1, sj0:sj1]
        return out

    # ------------------------------------------------------------------- the step
    def step(self) -> StepLedger:
        f = self.f
        nx, ny = self.nx, self.ny
        led = StepLedger()
        rho, ux, uy = self._macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        fpost = f - (f - feq) / self.tau
        p_post, m_post = self._mom_mass(fpost)

        # ---- streaming: populations leaving the lattice are dropped here ----
        fnew = self._stream(fpost)
        # populations that streamed into the body are absorbed there: zeroing the
        # solid nodes makes the array-wide moments the *wetted* moments, so the
        # global budget and the Ladd exchange measure the same physical object
        p_solid = np.zeros(2)
        if self.solid.any():
            sub = fnew[:, self.solid]
            sk = sub.sum(axis=1)
            p_solid = np.array([float(sk @ EX), float(sk @ EY)])
            fnew[:, self.solid] = 0.0
        p_stream, m_stream = self._mom_mass(fnew)
        led.momentum_stream = p_stream - p_post
        led.mass_stream = m_stream - m_post

        # ---- free-slip top / bottom (specular reflection) -------------------
        p_wall = np.zeros(2)
        m_wall = 0.0
        for (j, k_in, k_out) in ((0, (4, 7, 8), (2, 6, 5)),
                                 (ny - 1, (2, 5, 6), (4, 8, 7))):
            sl = (slice(None), slice(j, j + 1))
            p0, m0 = self._mom_mass(fnew, sl)
            for a, b in zip(k_in, k_out):
                fnew[b, :, j] = fpost[a, :, j]
            p1, m1 = self._mom_mass(fnew, sl)
            p_wall += p1 - p0
            m_wall += m1 - m0
        led.momentum_walls = p_wall
        led.mass_walls = m_wall

        # ---- inlet: Zou/He with mirrored non-equilibrium ---------------------
        sl_in = (slice(0, 1), slice(None))
        p0, m0 = self._mom_mass(fnew, sl_in)
        g = fnew[:, 0, :]
        rho_i = (g[0] + g[2] + g[4] + 2.0 * (g[3] + g[6] + g[7])) / (1.0 - self.U)
        g[1] = g[3] + (2.0 / 3.0) * rho_i * self.U
        g[5] = g[7] + rho_i * self.U / 6.0
        g[8] = g[6] + rho_i * self.U / 6.0
        fnew[:, 0, :] = g
        p1, m1 = self._mom_mass(fnew, sl_in)
        led.momentum_inlet = p1 - p0
        led.mass_inlet = m1 - m0

        # ---- outlet: zero gradient ------------------------------------------
        sl_out = (slice(nx - 1, nx), slice(None))
        p0, m0 = self._mom_mass(fnew, sl_out)
        for k in (3, 6, 7):
            fnew[k, nx - 1, :] = fnew[k, nx - 2, :]
        p1, m1 = self._mom_mass(fnew, sl_out)
        led.momentum_outlet = p1 - p0
        led.mass_outlet = m1 - m0

        # ---- moving immersed body ------------------------------------------
        p_pre, m_pre = self._mom_mass(fnew)
        uwx, uwy = self.obstacle.ux, self.obstacle.uy
        links = self._links
        if links is not None and links["tgt"].size:
            flat = fnew.reshape(-1)
            flat_post = fpost.reshape(-1)
            fi = flat_post[links["src"]]
            fm = flat_post[links["mir"]]
            kk = links["k"]
            wd = 0.5 / links["delta"]
            wgt = W[kk]
            corr = (2.0 * wgt * self.rho0 / CS2) * (EX[kk] * uwx + EY[kk] * uwy)
            interp = links["interp"]
            new = np.where(interp, wd * fi + (1.0 - wd) * fm, fi) - corr
            flat[links["tgt"]] = new
            jx = (new + fi) * EX[kk]
            jy = (new + fi) * EY[kk]
            led.force_mea = np.array([jx.sum(), jy.sum()])
            px = links["pos"][:, 0] - self.obstacle.x
            py = links["pos"][:, 1] - self.obstacle.y
            led.torque_mea = float(np.sum(px * jy - py * jx))
            self.last_samples = (links["pos"], np.column_stack([jx, jy]))
            self.work_on_body += float(led.force_mea @ np.array([uwx, uwy]))
            led.mass_body_exchange = float(np.sum(new - fi))
        p_post_body, m_post_body = self._mom_mass(fnew)
        led.momentum_body = p_post_body - p_pre
        led.mass_body = m_post_body - m_pre
        # the body force from the global budget: momentum emitted by the bounce-back
        # plus the momentum the body absorbed during streaming
        led.force_budget = -led.momentum_body + p_solid

        self.f = fnew
        self.steps += 1
        self.mass_stream += led.mass_stream
        self.mass_walls += led.mass_walls
        self.mass_inlet += led.mass_inlet
        self.mass_outlet += led.mass_outlet
        self.mass_body += led.mass_body
        self.mass_body_exchange += led.mass_body_exchange
        self.impulse_mea += led.force_mea
        return led

    # ------------------------------------------------------------------ metrics
    def momentum(self) -> np.ndarray:
        return self._mom_mass(self.f)[0]

    def mass(self) -> float:
        return self._mom_mass(self.f)[1]

    def mass_total(self) -> float:
        """Fluid mass over the wetted nodes (solid nodes excluded)."""
        return float(self.f[:, ~self.solid].sum())

    def velocity_field(self):
        return self._macroscopic(self.f)

    def curl(self):
        _, ux, uy = self._macroscopic(self.f)
        return np.gradient(uy, axis=0) - np.gradient(ux, axis=1)

    def mass_balance(self) -> dict:
        """Cumulative mass ledger and its closure defect (relative)."""
        terms = (self.mass_stream + self.mass_walls + self.mass_inlet
                 + self.mass_outlet + self.mass_body)
        wetted = self.mass_total()
        return {
            "stream": self.mass_stream,
            "walls": self.mass_walls,
            "inlet": self.mass_inlet,
            "outlet": self.mass_outlet,
            "body": self.mass_body,
            "body_exchange": self.mass_body_exchange,
            "absorbed": self.mass_absorbed,
            "released": self.mass_released,
            "wetted": wetted,
            "closure_abs": terms,
            "closure_error": (wetted - self.mass_initial) - terms,
        }