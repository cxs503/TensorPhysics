"""Structured Q4 boundary strips for thermal verification examples."""
import numpy as np


def boundary_pairs(nx,ny):
    cells = np.arange(nx*ny).reshape(ny,nx)
    return ((3,cells[:,0],cells[:,1]),(1,cells[:,-1],cells[:,-2]),
             (0,cells[0],cells[1]),(2,cells[-1],cells[-2]))


def cubic_patches(outer,inner):
    # Center a three-cell tangential strip wherever the endpoint permits.
    # Keep the target element first, as required by the patch API.
    index = np.arange(len(outer))
    strip = np.clip(index-1,0,len(outer)-3)[:,None]+np.arange(3)
    block = (outer[strip][:,:,None]+(inner-outer)[strip][:,:,None]*np.arange(3)).reshape(len(outer),9)
    return np.column_stack((outer,block[block!=outer[:,None]].reshape(len(outer),8)))
