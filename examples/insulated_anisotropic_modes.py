"""Complex low-frequency diffusion modes on an insulated anisotropic unit square."""
import json

import numpy as np
import torch

from conduction_provenance import provenance
from tensorfvm.cht import energy
from tensordg import poisson_2d_bc as face, transient_diffusion_2d as dg


KX, KY, CAPACITY = 2., .7, 1.5
MODES = ((1,2,1.+.2j),(2,1,-.3+.8j))


def forcing(x,y):
    return sum(a*torch.cos(i*torch.pi*x)*torch.cos(j*torch.pi*y) for i,j,a in MODES)


def reference(nx,ny):
    # Means along a face include the tangential sinc factor. A cell mean
    # averages both cosines. The lag coefficient for eigenvalue lambda is
    # -capacity/lambda times the steady coefficient.
    x,y = np.arange(nx+1)/nx,np.arange(ny+1)/ny
    result = [np.zeros(shape,complex) for shape in
              ((nx,ny),(nx,ny),(nx,ny),(nx+1,ny),(nx,ny+1))]
    source,steady,lag,qx,qy = result
    for i,j,amp in MODES:
        u,v = i*np.pi,j*np.pi
        lam = KX*u*u+KY*v*v
        cx = np.cos(u*(x[:-1]+.5/nx))*np.sinc(i/(2*nx))
        cy = np.cos(v*(y[:-1]+.5/ny))*np.sinc(j/(2*ny))
        shape = np.outer(cx,cy)
        first = -CAPACITY*amp/lam**2
        source += amp*shape
        steady += amp*shape/lam
        lag += first*shape
        qx += KX*u*first*np.outer(np.sin(u*x),cy)
        qy += KY*v*first*np.outer(cx,np.sin(v*y))
    return result


def measure(n):
    nx,ny = n,n+n//2
    source,steady,lag,qx,qy = reference(nx,ny)
    bc = dict.fromkeys(("left","right","bottom","top"),"neumann")
    for name,degree in (("FVM",0),("DG-P1",1),("DG-P2",2)):
        if degree:
            _,_,u,v = dg.solve_low_frequency_diffusion_2d(ncells=(nx,ny),degree=degree,
                forcing=forcing,bc=bc,conductivity=(KX,KY),capacity=CAPACITY,complex_load=True)
            fx,fy = face.poisson_face_powers_2d(v,bc=bc,conductivity=(KX,KY))
            u,v,fx,fy = u[:,:,0,0].numpy(),v[:,:,0,0].numpy(),fx.numpy(),fy.numpy()
        else:
            dc,first = energy.solve_low_frequency_conduction(np.full(nx,1/nx),np.full(ny,1/ny),
                KX,CAPACITY,{},source=source.T,conductivity_y=KY,return_interior_power=True)
            u,v = dc[0].T,first[0].T
            wall,interior = first[1:]
            fx = np.column_stack((wall["west"],interior["x"],-wall["east"])).T
            fy = np.vstack((wall["south"],interior["y"],-wall["north"])).T
        defect = CAPACITY*u/(nx*ny)+np.diff(fx,axis=0)+np.diff(fy,axis=1)
        yield dict(solver=name,nx=nx,ny=ny,max_dc_error_K=float(abs(u-steady).max()),
                   max_lag_error_K_s=float(abs(v-lag).max()),
                   x_flux_lag_error_W_s_m2=float(abs(ny*fx-qx).max()),
                   y_flux_lag_error_W_s_m2=float(abs(nx*fy-qy).max()),
                   max_cell_lag_power_residual_W_s=float(abs(defect).max()),
                   dc_energy_J=float(abs(CAPACITY*u.mean())),lag_energy_J_s=float(abs(CAPACITY*v.mean())))


if __name__ == "__main__":
    rows,previous = [],{}
    for n in (4,8,16,32):
        for row in measure(n):
            name = row["solver"]
            keys = ("max_dc_error_K","max_lag_error_K_s","x_flux_lag_error_W_s_m2","y_flux_lag_error_W_s_m2")
            if name in previous:
                row["orders"] = {k:float(np.log2(previous[name][k]/row[k])) for k in keys}
            row["passed"] = max(row["max_cell_lag_power_residual_W_s"],row["dc_energy_J"],row["lag_energy_J_s"]) < 1e-11
            if n == 32:
                quadratic = name == "DG-P2"
                row["passed"] &= (min(row["orders"].values()) > (3.7 if quadratic else 1.8)
                                  and row["max_dc_error_K"] < (1e-8 if quadratic else 5e-5)
                                  and row["max_lag_error_K_s"] < (1e-9 if quadratic else 3e-6)
                                  and max(row[keys[2]],row[keys[3]]) < (2e-9 if quadratic else 1e-5))
            previous[name] = row
            rows.append(row)
    print(json.dumps(dict(runs=rows,sources=[provenance(f) for f in (__file__,dg.__file__,face.__file__,energy.__file__)]),
                     indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
