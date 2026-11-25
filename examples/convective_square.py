"""Steady unit-conductivity square with prescribed ambient temperatures on four films."""
import argparse
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordg import poisson_2d_bc as dg
from tensorfvm.cht import energy


def measure(n, coarse_fvm):
    width = 1/n
    center = (np.arange(n)+.5)*width
    ix = np.exp(center)*np.sinh(width/2)/(width/2)
    iy = np.exp(.5*center)*np.sinh(width/4)/(width/4)
    exact = ix[:, None]*iy  # x index first, square-cell means
    trace = np.stack((iy, np.e*iy, ix, np.exp(.5)*ix))
    outward = np.array([1., -1., .5, -.5])[:, None]*trace
    films = [2., 5., 3., 7.]
    ambient = trace-outward/np.array(films)[:,None]
    source = -1.25*exact
    fvm_bc = {side: dict(type="robin", values=amb, film_coefficient=h)
              for side,amb,h in zip(("west","east","south","north"), ambient, films)}
    widths = np.full(n,width)
    if n >= 6 or coarse_fvm:
        t, info = energy.solve_steady_energy(widths, widths, np.ones((n,n)), np.zeros((n,n+1)),
                           np.zeros((n+1,n)), fvm_bc, source=source.T, high_order=False, return_wall_power=True)
        flux = -np.stack([info["wall_power"][s] for s in ("west","east","south","north")])/width
        yield "FVM", t.T, ambient+flux/np.array(films)[:,None], flux, info["ledger"].residual_inf, exact, trace, outward
    if n > 24:
        return
    sides = ("left","right","bottom","top")
    boundaries = dict.fromkeys(sides, "robin")
    data = {side:(h, lambda x,y,factor=1-sign/h: factor*torch.exp(x+.5*y))
            for side,h,sign in zip(sides,films,(1.,-1.,.5,-.5))}
    for degree in (1,2,3):
        _, _, coeff, _ = dg.solve_poisson_2d_bc(n, degree, sparse=True, bc=boundaries, robin_films=data,
                                    forcing=lambda x,y: -1.25*torch.exp(x+.5*y))
        fx, fy = dg.poisson_face_powers_2d(coeff, bc=boundaries, robin_films=data)
        flux = np.stack((-fx[0],fx[-1],-fy[:,0],fy[:,-1]))/width
        # Legendre modes of positive tangential degree integrate to zero.
        u = coeff.numpy()
        minus = (-1.)**np.arange(degree+1)
        wall_t = np.stack((u[0,:,:,0] @ minus, u[-1,:,:,0].sum(axis=1),
                           u[:,0,0,:] @ minus, u[:,-1,0,:].sum(axis=1)))
        balance = abs(fx.diff(dim=0).numpy()+fy.diff(dim=1).numpy()-source*width**2).max()
        yield f"DG-P{degree}", u[:,:,0,0], wall_t, flux, balance, exact, trace, outward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-fvm", action="store_true", help="include the three-cell FVM control grid")
    coarse = parser.parse_args().coarse_fvm
    runs, previous = [], {}
    for n in (3,6,12,24,48,96):
        for name, t, wall, q, balance, exact, trace, power in measure(n, coarse):
            errors = np.array([abs(t-exact).max(), abs(wall-trace).max(), abs(q-power).max()])
            rates = np.log2(previous[name]/errors) if name in previous else None
            passed = np.isfinite(errors).all() and balance < 1e-10
            if rates is not None:
                passed &= min(rates) > 1.7
            runs.append(dict(solver=name, cells_per_side=n, max_cell_mean_error_K=errors[0],
                             max_wall_mean_error_K=errors[1], max_wall_flux_error_W_m2=errors[2],
                             max_cell_balance_W=balance, orders=None if rates is None else rates.tolist(),
                             passed=bool(passed)))
            previous[name] = errors
    print(json.dumps(dict(runs=runs,sources=[provenance(f) for f in (__file__,dg.__file__,energy.__file__)]),
                     indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
