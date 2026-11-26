"""Analytical decaying contact eigenmode shared by the rod and slab benchmarks."""
import numpy as np


def material_properties():
    # The first layer spans a quarter of pi in phase, the second three quarters.
    # The outer Dirichlet values vanish. Choose material properties so both
    # pieces decay at the same rate and transmit the same rightward flux.
    interface, decay = .4, 3.
    wl, wr = np.pi/(4*interface), 3*np.pi/(4*(1-interface))
    conductivity = np.array([1., wl/(2*wr)])
    capacity = conductivity*np.array([wl, wr])**2/decay
    resistance = 1/wl
    return interface, decay, conductivity, capacity, resistance
