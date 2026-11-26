"""Partition affine internal heating between two isothermal walls across a resistive contact."""
import argparse
import json

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq
from tensorsph import heat_conduction_2d as sph

from conduction_provenance import provenance


def measure(n, sign, resistance, balanced=False, *, fvm=False):
    cut = 3*n//8
    lengths = (cut/n, 1-cut/n)
    conductivities = (1.3, 3.7)
    means = (1.1, -1.1*lengths[0]/lengths[1]) if balanced else (1.1, .6)
    slopes = (sign*4., -sign*2.)
    # Start at zero west flux. The east temperature residual determines
    # the uniform flux correction through the full series resistance.
    q0, t0 = 0., 0.
    sources, fluxes, temperatures = [], [], []
    for i, (width, k, mean, slope) in enumerate(zip(lengths, conductivities, means, slopes)):
        source = Polynomial([mean-slope*width/2, slope])
        flux = source.integ(k=q0)
        temperature = (-flux/k).integ(k=t0)
        sources.append(source)
        fluxes.append(flux)
        temperatures.append(temperature)
        q0 = flux(width)
        t0 = temperature(width)
        if i == 0:
            t0 -= resistance*q0
    correction = t0/(sum(h/k for h, k in zip(lengths, conductivities))+resistance)
    local_faces = [np.arange(m+1)/n for m in (cut, n-cut)]
    centers = [(x[1:]+x[:-1])/2 for x in local_faces]
    path = [centers[0]/conductivities[0],
            lengths[0]/conductivities[0]+resistance+centers[1]/conductivities[1]]
    exact_t = np.concatenate([t(x)-correction*r for t, x, r in zip(temperatures, centers, path)])
    exact_q = correction+np.r_[fluxes[0](local_faces[0][:-1]), fluxes[1](local_faces[1])]
    np.testing.assert_allclose(exact_q[-1]-exact_q[0], np.dot(means, lengths), atol=1e-14, rtol=0.)
    source = np.concatenate([s(x) for s, x in zip(sources, centers)])
    contacts = np.zeros(n-1)
    contacts[cut-1] = resistance
    k = np.repeat(conductivities, [cut, n-cut])
    if fvm:
        from tensorfvm.cht import rod

        field, _, west, east, (left_trace, right_trace) = rod.solve_steady_stepped_rod(
            np.full(n, 1/n), 1., k, 1., (0., 0.), source=source,
            contact_resistance=contacts, flux_order=2, source_derivative_breaks=[cut], return_traces=True)
        np.testing.assert_allclose(east[:-1], west[1:], rtol=0., atol=1e-10)
        q = np.r_[west, east[-1]]
        # The exact average of a cubic adds T''(center)*dx²/24.
        exact_t -= source/(24*k*n**2)
    else:
        field, flux = sph.solve_harmonic_layered_slab(
            source, 1/n, (0., 0.), 0., conductivity=k, volumetric_heat_capacity=1.,
            contact_resistance_x=contacts, return_x_flux=True, backend="banded")
        left_trace, right_trace = sph.layered_slab_contact_temperatures(
            field, flux, 1/n, k, axial_flux_curvature=np.repeat(slopes, [cut, n-cut]))
        left_trace, right_trace = left_trace.numpy(), right_trace.numpy()
        field, q = field.numpy(), flux.numpy()
    traces = np.array([left_trace[cut-1], right_trace[cut-1]])
    left_exact = temperatures[0](lengths[0])-correction*lengths[0]/conductivities[0]
    right_exact = temperatures[1](0.)-correction*(lengths[0]/conductivities[0]+resistance)
    jump_error = abs(traces[0]-traces[1]-resistance*q[cut])
    row = dict(columns=n, gradient_sign=sign, resistance_m2_K_W=resistance,
                max_contact_trace_error_K=float(abs(traces-[left_exact, right_exact]).max()),
                contact_jump_residual_K=float(jump_error),
                west_outward_W_m2=float(-q[0].real), east_outward_W_m2=float(q[-1].real),
                temperature_error_K=float(abs(field-exact_t).max()),
                wall_flux_error_W_m2=float(abs((q-exact_q)[[0,-1]]).max()),
                all_face_flux_error_W_m2=float(abs(q-exact_q).max()),
                max_cell_balance_W_m2=float(abs(np.diff(q)-source/n).max()))
    if balanced:
        # Both walls have the same temperature and the net source vanishes.
        # Each layer has a fixed source sign, so a flux reversal is unique there.
        roots = []
        quadratic_roots = []
        exact_roots = []
        extrema, exact_extrema = [], []
        for i, (width, polynomial, face_x) in enumerate(zip(lengths, fluxes, local_faces)):
            origin = 0. if i == 0 else lengths[0]
            indices = slice(0, cut+1) if i == 0 else slice(cut, None)
            numerical = q[indices].real
            direction = np.sign(means[i])
            assert np.all(direction*np.diff(numerical) > 0)
            assert numerical[0]*numerical[-1] < 0
            roots.append(origin+np.interp(0., direction*numerical, face_x))
            # Recover q within the bracketing cell from its numerical face
            # values and the slope of the sampled source inside this material.
            layer_source = source[:cut] if i == 0 else source[cut:]
            dx = 1/n
            curvature = .5*dx*(layer_source[-1]-layer_source[0])/(len(layer_source)-1)
            cell = np.searchsorted(direction*numerical, 0.)-1
            left, right = numerical[cell:cell+2]
            def local_flux(f):
                return (1-f)*left+f*right+curvature*f*(f-1)
            fraction = brentq(local_flux, 0., 1.)
            quadratic_roots.append(origin+face_x[cell]+dx*fraction)
            valid = [v.real for v in (polynomial+correction).roots() if v.imag == 0 and 0 < v.real < width]
            assert len(valid) == 1
            exact_roots.append(origin+valid[0])

            # Integrate Fourier's law in this cell. Anchor the primitive at
            # its mean for an FVM cell or its midpoint for an SPH particle.
            primitive = Polynomial([left, right-left-curvature, curvature]).integ()
            anchor = primitive.integ()(1.) if fvm else primitive(.5)
            index = cell if i == 0 else cut+cell
            cond = conductivities[i]
            extrema.append(float(field[index].real-dx*(primitive(fraction)-anchor)/cond))
            reference_path = valid[0]/cond
            if i == 1:
                reference_path += lengths[0]/conductivities[0]+resistance
            exact_extrema.append(temperatures[i](valid[0])-correction*reference_path)
        row.update(wall_balance_W_m2=float(abs(q[-1]-q[0])),
                   hot_and_cold_temperatures_K=extrema, analytical_hot_and_cold_K=exact_extrema,
                   max_extremum_temperature_error_K=float(np.max(np.abs(np.subtract(extrema, exact_extrema)))),
                   flux_reversals_m=roots, analytical_flux_reversals_m=exact_roots,
                   quadratic_flux_reversals_m=quadratic_roots,
                   max_quadratic_reversal_error_m=float(np.max(np.abs(np.subtract(quadratic_roots, exact_roots)))),
                   max_reversal_error_m=float(np.max(np.abs(np.subtract(roots, exact_roots)))))
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced", action="store_true", help="balance layer heating with cooling and locate both flux reversals")
    parser.add_argument("--compare-fvm", action="store_true", help="also verify FVM cell means and numerical face powers for unit area")
    args = parser.parse_args()
    balanced = args.balanced
    paths = [__file__, sph.__file__]
    pythonpath = "../TensorSPH/src"
    if args.compare_fvm:
        from tensorfvm.cht import rod

        paths.append(rod.__file__)
        pythonpath += ":../TensorFVM/src"
    runs = []
    for sign in (-1., 1.):
        for resistance in (0., .25):
            previous = None
            previous_root_error = None
            previous_extremum_error = None
            for n in (16, 32, 64, 128):
                row = measure(n, sign, resistance, balanced)
                errors = np.array([row["temperature_error_K"], row["wall_flux_error_W_m2"]])
                orders = np.log2(previous/errors) if previous is not None else None
                row["temperature_and_wall_flux_orders"] = orders.tolist() if orders is not None else None
                assert row["max_cell_balance_W_m2"] < 1e-10
                if balanced:
                    assert row["wall_balance_W_m2"] < 1e-10
                    assert row["west_outward_W_m2"] > 0 > row["east_outward_W_m2"]
                    hot, cold = row["hot_and_cold_temperatures_K"]
                    assert hot > 0 > cold
                    extremum_error = row["max_extremum_temperature_error_K"]
                    if previous_extremum_error is not None:
                        row["extremum_temperature_order"] = float(np.log2(previous_extremum_error/extremum_error))
                    previous_extremum_error = extremum_error
                    root_error = row["max_quadratic_reversal_error_m"]
                    if previous_root_error is not None:
                        row["quadratic_reversal_order"] = float(np.log2(previous_root_error/root_error))
                    previous_root_error = root_error
                    if n == 128:
                        assert row["quadratic_reversal_order"] > 1.8
                        assert row["extremum_temperature_order"] > 1.8
                        assert extremum_error < 1.5e-5
                        assert root_error < 1.1e-5
                        assert row["max_reversal_error_m"] < 2e-4
                if n == 128:
                    np.testing.assert_array_less(1.7, orders)
                    assert row["max_contact_trace_error_K"] < 2e-5
                    assert row["contact_jump_residual_K"] < 2.5e-5
                    assert row["all_face_flux_error_W_m2"] < 1e-5
                    assert row["temperature_error_K"] < 2e-5
                previous = errors
                runs.append(row)
                if args.compare_fvm:
                    fv = measure(n, sign, resistance, balanced, fvm=True)
                    row["solver"], fv["solver"] = "SPH", "FVM"
                    for metric in ("temperature_error_K", "all_face_flux_error_W_m2", "max_cell_balance_W_m2",
                                   "max_contact_trace_error_K", "contact_jump_residual_K"):
                        assert fv[metric] < 1e-10
                    if balanced:
                        assert fv["wall_balance_W_m2"] < 1e-10
                        assert fv["west_outward_W_m2"] > 0 > fv["east_outward_W_m2"]
                        assert fv["max_extremum_temperature_error_K"] < 1e-10
                        assert fv["max_quadratic_reversal_error_m"] < 1e-10
                        if n == 128:
                            assert fv["max_reversal_error_m"] < 2e-4
                        fv["max_reversal_difference_from_SPH_m"] = float(np.max(np.abs(
                            np.subtract(row["flux_reversals_m"], fv["flux_reversals_m"]))))
                    runs.append(fv)
    print(json.dumps(dict(reference="Piecewise cubic Fourier temperatures with two zero-temperature walls",
                          reproduction=f"OMP_NUM_THREADS=1 PYTHONPATH={pythonpath} python examples/sph_affine_contact_wall_heat.py"+(" --balanced" if balanced else "")+(" --compare-fvm" if args.compare_fvm else ""),
                          sources=[provenance(p) for p in paths], runs=runs),
                     indent=2, allow_nan=False))
