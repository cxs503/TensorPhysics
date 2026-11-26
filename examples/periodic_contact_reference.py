"""Continuum reference for a unit wall-temperature mode across two conducting layers."""
import numpy as np


def reference(n, resistance, omega, transverse_wavenumber=0., *,
              conductivity=(1., 4.), capacity=(1., 2.), return_face_flux=False):
    """Propagate the continuum (temperature, rightward flux) from the cold wall."""
    waves = np.sqrt(transverse_wavenumber**2 + 1j*omega*np.asarray(capacity)/np.asarray(conductivity))

    def propagate(length, layer, temperature, flux):
        z = length*waves[layer]
        admittance = conductivity[layer]*waves[layer]
        return (np.cosh(z)*temperature + np.sinh(z)*flux/admittance,
                admittance*np.sinh(z)*temperature + np.cosh(z)*flux)

    # Start with unit flux at x=1, then normalize the left boundary to one kelvin.
    cold_trace, contact_flux = propagate(.5, 1, 0., 1.)
    hot_trace = cold_trace + resistance*contact_flux
    scale, hot_flux = propagate(.5, 0, hot_trace, contact_flux)
    back = np.linspace(.5, 0., n//2+1)
    left, left_flux = propagate(back, 0, hot_trace, contact_flux)
    right, right_flux = propagate(back, 1, 0., 1.)
    nodes = np.r_[left, right]/scale
    # On a uniform cell the average is the midpoint value times sinh(z)/z.
    z = waves/(2*n)
    back_mid = .5*(back[:-1]+back[1:])
    avg = np.concatenate([propagate(back_mid, layer, temp, q)[0]*np.sinh(z[layer])/z[layer]
                          for layer, temp, q in ((0, hot_trace, contact_flux), (1, 0., 1.))])/scale
    result = (nodes, avg, contact_flux/scale, np.array([hot_flux, -1.])/scale)
    if return_face_flux:
        return (*result, np.r_[left_flux[:-1], right_flux]/scale)
    return result
