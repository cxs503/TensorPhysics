"""Two conducting layers separated by a lumped thermal sheet and two equal contact resistances."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensordem.thermal import contact_chain
from tensorfem import thermal


def reference(omega, resistance, sheet_capacity):
    # Propagate temperature and rightward flux from the cold end (T=0,q=1).
    # The final normalization sets the hot end to a unit temperature phasor.
    def backward(k, capacity, temperature, flux):
        wave = np.sqrt(1j*omega*capacity/k)
        c, s = np.cosh(.5*wave), np.sinh(.5*wave)
        return c*temperature+s*flux/(k*wave), k*wave*s*temperature+c*flux

    cold, outgoing = backward(4., 2., 0., 1.)
    sheet = cold + .5*resistance*outgoing
    incoming = outgoing + 1j*omega*sheet_capacity*sheet
    hot = sheet + .5*resistance*incoming
    wall, _ = backward(1., 1., hot, incoming)
    return np.array([sheet, incoming, outgoing])/wall


def measure(n, omega, resistance, sheet_capacity):
    mid = n//2
    dx = np.full(n, 1/n)
    k = np.repeat([1., 4.], mid)
    cv = np.repeat([1., 2.], mid)
    contacts, sheet = np.zeros(n-1), np.zeros(n-1)
    contacts[mid-1], sheet[mid-1] = resistance, sheet_capacity
    r, c = contact_chain.cell_centered_slab_network(dx, k, cv, contact_resistance=contacts,
                                                    interface_heat_capacity=sheet)
    t = contact_chain.harmonic_contact_temperatures(r, c, omega, t_left=1., t_right=0.)
    q = -np.diff(t)/r
    dem = np.array([t[mid+1], q[mid], q[mid+1]])

    # Two independent LINE2 meshes, plus one global sheet node between the traces.
    x = np.linspace(0., 1., n+1)
    nodes = torch.from_numpy(np.r_[x[:mid+1], x[mid:], .5][:, None])
    left = torch.arange(n)
    left[mid:] += 1
    ends = torch.tensor([0, n+1])
    sheet_id = n+2
    lump = nodes.new_zeros(n+3)
    lump[sheet_id] = sheet_capacity
    model = thermal.ThermalModel(nodes, torch.stack((left, left+1), 1), torch.from_numpy(k),
                    1., torch.from_numpy(cv), ends, nodes.new_tensor([1., 0.]), nodal_heat_capacity=lump,
                    contact_pairs=torch.tensor([[mid, sheet_id], [sheet_id, mid+1]]),
                    contact_conductance=2/resistance)
    sol = thermal.solve_harmonic_thermal(model, omega, sparse=True)
    power = thermal.thermal_element_nodal_power(model, sol.temperature, omega).numpy()
    fem = np.array([sol.temperature[sheet_id].item(), -power[mid-1, 1], power[mid, 0]])
    return {"DEM": dem, "FEM": fem}


def main():
    rows = []
    for omega, capacity in ((.75, .12), (8., .8)):
        resistance = .3
        exact = reference(omega, resistance, capacity)
        previous = {}
        for n in (8, 16, 32, 64, 128):
            for name, observed in measure(n, omega, resistance, capacity).items():
                errors = abs(observed-exact)
                balance = abs(observed[1]-observed[2]-1j*omega*capacity*observed[0])
                orders = np.log2(previous[name]/errors) if name in previous else None
                passed = np.isfinite(errors).all() and balance < 1e-11
                if orders is not None:
                    passed &= min(orders) > 1.8
                rows.append(dict(solver=name, n=n, omega_rad_s=omega, sheet_capacity_J_m2K=capacity,
                                 sheet_temperature_error_K=errors[0], left_flux_error_W_m2=errors[1],
                                 right_flux_error_W_m2=errors[2], sheet_balance_W_m2=balance,
                                 orders=orders.tolist() if orders is not None else None, passed=bool(passed)))
                previous[name] = errors
    print(json.dumps(dict(resistance_m2K_W=resistance, runs=rows,
                         sources=[provenance(f) for f in (__file__, thermal.__file__, contact_chain.__file__)]),
                     indent=2, allow_nan=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
