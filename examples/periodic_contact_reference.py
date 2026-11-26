"""Continuum reference for a unit wall-temperature mode across two conducting layers."""
import numpy as np


def reference(n, resistance, omega, transverse_wavenumber=0., *,
              conductivity=(1., 4.), capacity=(1., 2.), return_face_flux=False, contact_face=None):
    """Propagate the continuum (temperature, rightward flux) from the cold wall."""
    cut = n//2 if contact_face is None else contact_face
    if not 0 < cut < n:
        raise ValueError("contact must be an internal face")
    width = cut/n
    waves = np.sqrt(transverse_wavenumber**2 + 1j*omega*np.asarray(capacity)/np.asarray(conductivity))

    def sinhc(z):
        # np.sinc includes the removable singularity at zero.
        return np.sinc(z/(1j*np.pi))

    def propagate(length, layer, temperature, flux):
        z = length*waves[layer]
        admittance = conductivity[layer]*waves[layer]
        return (np.cosh(z)*temperature + length*sinhc(z)*flux/conductivity[layer],
                admittance*np.sinh(z)*temperature + np.cosh(z)*flux)

    # Start with unit flux at x=1, then normalize the left boundary to one kelvin.
    cold_trace, contact_flux = propagate(1-width, 1, 0., 1.)
    hot_trace = cold_trace + resistance*contact_flux
    scale, hot_flux = propagate(width, 0, hot_trace, contact_flux)
    left_back = np.linspace(width, 0., cut+1)
    right_back = np.linspace(1-width, 0., n-cut+1)
    left, left_flux = propagate(left_back, 0, hot_trace, contact_flux)
    right, right_flux = propagate(right_back, 1, 0., 1.)
    nodes = np.r_[left, right]/scale
    # On a uniform cell the average is the midpoint value times sinh(z)/z.
    z = waves/(2*n)
    avg = np.concatenate([propagate(.5*(back[:-1]+back[1:]), layer, temp, q)[0]*sinhc(z[layer])
                          for layer, back, temp, q in ((0, left_back, hot_trace, contact_flux), (1, right_back, 0., 1.))])/scale
    result = (nodes, avg, contact_flux/scale, np.array([hot_flux, -1.])/scale)
    if return_face_flux:
        return (*result, np.r_[left_flux[:-1], right_flux]/scale)
    return result
