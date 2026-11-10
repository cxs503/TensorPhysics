"""Optional compiled streaming/collision kernel for the D2Q9 cylinder solver.

The pure-NumPy path in :mod:`~tensorphysics.lbm_bem_cross.lbm` is written for
clarity and portability (it is the reference implementation used by the unit
tests) but it is memory-bandwidth bound by its many temporary arrays: at
``300k`` lattice nodes it runs at ~170 ms/step.  This module compiles a small C
kernel (``cc -O3``) that fuses collision, streaming, half-way bounce-back and
the momentum-exchange force into a single pass, which is ~15x faster and makes
the ``Re = 1e4`` campaign runs feasible.

The kernel is *optional*: if no C compiler is available the module reports
``available() == False`` and the caller silently falls back to NumPy — the
physics is identical (verified by ``tests/test_lbm_bem_cross.py``, which checks
that both backends produce the same fields to float32 round-off).
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["available", "step_kernel", "build_error"]

_SRC = r"""
#include <math.h>
#include <stddef.h>

static const int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
static const float W[9] = {4.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
                           1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f};

/* One lattice time step: moments -> regularised BGK collision -> streaming with
 * half-way bounce-back on solid links -> momentum-exchange force.
 * f and g are (9, ny, nx) arrays; the result is written into g. */
void lbm_step(const float* __restrict__ f, float* __restrict__ g,
              const unsigned char* __restrict__ solid,
              float* __restrict__ rho_out, float* __restrict__ u_out,
              float* __restrict__ v_out,
              int ny, int nx, float beta, float tau, float cs_les,
              double* force)
{
    const float cs2 = 1.f/3.f;
    const size_t plane = (size_t)ny * (size_t)nx;
    double fx = 0.0, fy = 0.0;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const size_t idx = (size_t)iy * nx + ix;
            if (solid[idx]) { continue; }
            float fi[9];
            float rho = 0.f;
            for (int i = 0; i < 9; ++i) { fi[i] = f[i*plane + idx]; rho += fi[i]; }
            const float inv = 1.f / rho;
            float ux = (fi[1] + fi[5] + fi[8] - fi[3] - fi[6] - fi[7]) * inv;
            float uy = (fi[2] + fi[5] + fi[6] - fi[4] - fi[7] - fi[8]) * inv;
            rho_out[idx] = rho; u_out[idx] = ux; v_out[idx] = uy;

            const float Pxx = (fi[1]+fi[3]+fi[5]+fi[6]+fi[7]+fi[8]) - rho*(cs2 + ux*ux);
            const float Pyy = (fi[2]+fi[4]+fi[5]+fi[6]+fi[7]+fi[8]) - rho*(cs2 + uy*uy);
            const float Pxy = (fi[5]-fi[6]+fi[7]-fi[8]) - rho*ux*uy;

            float beta_l = beta;
            if (cs_les > 0.f) {
                const float Pn = sqrtf(Pxx*Pxx + Pyy*Pyy + 2.f*Pxy*Pxy);
                const float q = sqrtf(18.f*cs_les*cs_les*Pn/(rho*tau) + 1.f) - 1.f;
                beta_l = beta * (1.f - 0.5f*q) / (1.f + 0.5f*q);
            }
            const float usq = ux*ux + uy*uy;
            const float base = 1.f - 1.5f*usq;
            float post[9];
            for (int i = 0; i < 9; ++i) {
                const float cu = 3.f*(CX[i]*ux + CY[i]*uy);
                const float feq = W[i]*rho*(base + cu + 0.5f*cu*cu);
                const float Q = W[i]*(4.5f*((float)(CX[i]*CX[i]) - cs2)*Pxx
                                    + 4.5f*((float)(CY[i]*CY[i]) - cs2)*Pyy
                                    + 9.f*(float)(CX[i]*CY[i])*Pxy);
                post[i] = feq + beta_l*Q;
            }

            /* streaming: push into g; solid targets reflect, domain boundaries
             * reflect too but are overwritten by the boundary conditions */
            for (int i = 0; i < 9; ++i) {
                const int tx = ix + CX[i];
                const int ty = iy + CY[i];
                const int outside = (tx < 0 || tx >= nx || ty < 0 || ty >= ny);
                const int is_solid = (!outside) && solid[(size_t)ty*nx + tx] != 0;
                if (outside || is_solid) {
                    const int o = OPP[i];
                    if (o == i) { g[i*plane + idx] = post[i]; }
                    else { g[o*plane + idx] += post[i]; }
                    if (is_solid) {
                        fx += 2.0 * CX[i] * (double)post[i];
                        fy += 2.0 * CY[i] * (double)post[i];
                    }
                } else {
                    g[i*plane + (size_t)ty*nx + tx] = post[i];
                }
            }
        }
    }
    force[0] = fx; force[1] = fy;
}

void lbm_clear(float* g, int ny, int nx)
{
    const size_t n = (size_t)9 * (size_t)ny * (size_t)nx;
    for (size_t i = 0; i < n; ++i) g[i] = 0.f;
}
"""

_LIB = None
_ERROR: str | None = None


def _build() -> None:
    global _LIB, _ERROR
    if _LIB is not None or _ERROR is not None:
        return
    key = hashlib.sha1(_SRC.encode()).hexdigest()[:12]
    cache = Path(tempfile.gettempdir()) / "tensorphysics_lbm_kernel"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        so = cache / f"lbm_{key}.so"
        if not so.exists():
            cfile = cache / f"lbm_{key}.c"
            cfile.write_text(_SRC)
            proc = subprocess.run(
                ["cc", "-O3", "-shared", "-fPIC", "-o", str(so), str(cfile)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                _ERROR = f"cc failed: {proc.stderr[:400]}"
                return
        lib = ctypes.CDLL(str(so))
        fp = ctypes.c_void_p
        lib.lbm_step.argtypes = [fp, fp, fp, fp, fp, fp, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_float, ctypes.c_float, ctypes.c_float, fp]
        lib.lbm_step.restype = None
        lib.lbm_clear.argtypes = [fp, ctypes.c_int, ctypes.c_int]
        lib.lbm_clear.restype = None
        _LIB = lib
    except Exception as exc:  # pragma: no cover - depends on the host toolchain
        _ERROR = f"kernel build failed: {exc}"


def available() -> bool:
    """True when the compiled kernel could be built and loaded."""
    _build()
    return _LIB is not None


def build_error() -> str | None:
    _build()
    return _ERROR


def step_kernel(f: np.ndarray, g: np.ndarray, solid: np.ndarray, rho: np.ndarray,
                u: np.ndarray, v: np.ndarray, beta: float, tau: float,
                cs_les: float) -> tuple[float, float]:
    """Advance one step; ``f`` is read, ``g`` is overwritten (caller swaps them)."""
    if _LIB is None:
        raise RuntimeError("compiled kernel not available")
    ny, nx = solid.shape
    force = np.zeros(2, dtype=np.float64)
    _LIB.lbm_clear(g.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ny), ctypes.c_int(nx))
    _LIB.lbm_step(f.ctypes.data_as(ctypes.c_void_p), g.ctypes.data_as(ctypes.c_void_p),
                  solid.ctypes.data_as(ctypes.c_void_p),
                  rho.ctypes.data_as(ctypes.c_void_p), u.ctypes.data_as(ctypes.c_void_p),
                  v.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(ny), ctypes.c_int(nx),
                  ctypes.c_float(beta), ctypes.c_float(tau), ctypes.c_float(cs_les),
                  force.ctypes.data_as(ctypes.c_void_p))
    return float(force[0]), float(force[1])