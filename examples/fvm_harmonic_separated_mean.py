"""Verify tiny-frequency rod powers with bulk, end and split-contact storage."""
import json

import numpy as np

from conduction_provenance import provenance
from tensorfvm.cht import rod


def measure(n, omega, power):
    lengths = np.array([.37, .63])
    areas = np.array([.8, 1.4])
    k, c = np.array([.9, 3.7]), np.array([1.6, .85])
    widths = np.repeat(lengths/n, n)
    rate = .27+.34j
    drop_resistance = .32  # K/W over the contact area
    resistance = np.zeros(2*n-1)
    resistance[n-1] = drop_resistance*min(areas)
    # The larger (right) section owns the exposed shoulder.
    sheet, fraction, shoulder_capacity = .73, .38, .46
    trace_mass = np.array([sheet*fraction, sheet*(1-fraction)+shoulder_capacity*np.diff(areas)[0]])
    end_capacity = np.array([.3, .75])
    end_mass = areas*end_capacity
    west_contact = -power*lengths[0]/(k[0]*areas[0])
    east_contact = west_contact-drop_resistance*power
    faces_by_layer = [t-power*np.linspace(0., length, n+1)/(cond*area)
                      for t, length, cond, area in zip((0., east_contact), lengths, k, areas)]
    cells = np.concatenate([(f[:-1]+f[1:])/2 for f in faces_by_layer])
    ends = np.array([0., faces_by_layer[-1][-1]])
    traces = np.array([west_contact, east_contact])
    mass = widths*np.repeat(areas*c, n)
    shift = (mass @ cells+end_mass @ ends+trace_mass @ traces)/(mass.sum()+end_mass.sum()+trace_mass.sum())
    cells, ends, traces = cells-shift, ends-shift, traces-shift
    end_load = end_mass*(rate+1j*omega*ends)+np.array([power, -power])
    sheet_profile = np.zeros_like(resistance)
    sheet_profile[n-1] = sheet
    trace_loads = np.zeros((2, len(resistance)), dtype=complex)
    trace_loads[:, n-1] = trace_mass*(rate+1j*omega*traces)
    field, faces, west, east, sides, mean = rod.solve_harmonic_stepped_rod(
        widths, np.repeat(areas, n), np.repeat(k, n), np.repeat(c, n), end_load, omega,
        source=np.repeat(c, n)*(rate+1j*omega*cells), end_types=("power", "power"),
        end_areal_heat_capacity=end_capacity, shoulder_areal_heat_capacity=shoulder_capacity,
        interface_heat_capacity=sheet_profile, interface_capacity_left_fraction=fraction,
        contact_resistance=resistance, trace_power=trace_loads, flux_order=2,
        return_traces=True, separate_mean=True)
    return dict(cells_per_layer=n, omega_rad_s=omega, through_power_W=[power.real, power.imag],
                mean_rate_error_K_s=float(abs(1j*omega*mean-rate)),
                relative_temperature_error_K=float(max(max(abs(field-cells)), max(abs(faces[[0,-1]]-ends)),
                                                       max(abs(np.asarray(sides)[:, n-1]-traces)))),
                section_power_error_W=float(max(max(abs(west-power)), max(abs(east-power)))))


if __name__ == "__main__":
    rows = [measure(n, omega, power) for n in (16, 32, 64) for omega in (1., 1e-4, 1e-8, 1e-12)
            for power in (0j, .35-.21j)]
    for row in rows:
        row["passed"] = max(row["mean_rate_error_K_s"], row["relative_temperature_error_K"],
                            row["section_power_error_W"]) < 1e-9
    print(json.dumps(dict(reference="Uniform temperature pole plus piecewise linear temperature with constant axial power and a resistive jump",
                          reproduction="OPENBLAS_NUM_THREADS=1 PYTHONPATH=../TensorFVM/src python examples/fvm_harmonic_separated_mean.py",
                          runs=rows, sources=[provenance(f) for f in (__file__, rod.__file__)]), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
