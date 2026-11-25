"""FEM and boundary-map comparison of a heat-generating resistive cylindrical sheet."""
import argparse
import json
import math

import numpy as np
import torch

from conduction_provenance import provenance
from tensorbem import layered_annulus
from tensorfem import thermal, thermal_analytic


def measure(n,fraction,bulk=(0.,0.)):
    a,b,c = .3,.6,1.
    ki,ko,r,source = 2.,.8,.35,4.
    nt = 4*n
    circ = 2*math.pi*b
    ri = math.log(b/a)/(2*math.pi*ki)+fraction*r/circ
    ro = math.log(c/b)/(2*math.pi*ko)+(1-fraction)*r/circ
    si,so = bulk
    # Integrate d(2πr*q)/dr=2πr*S from the contact to each wall.
    pi_vol = math.pi*si*(b*b-a*a)
    po_vol = math.pi*so*(c*c-b*b)
    inner_offset = si/(2*ki)*(b*b*math.log(b/a)-(b*b-a*a)/2)
    outer_offset = so/(2*ko)*((c*c-b*b)/2-b*b*math.log(c/b))
    inner_rate = (inner_offset-outer_offset-circ*source*ro)/(ri+ro)
    sheet = inner_offset-inner_rate*ri
    exact_rates = np.array([inner_rate-pi_vol,inner_rate+circ*source+po_vol])
    rings = [thermal_analytic.annular_quad_mesh(n,nt,inner=lo,outer=hi,dtype=torch.float64)
             for lo,hi in ((a,b),(b,c))]
    count = len(rings[0].nodes)
    nodes = torch.cat([mesh.nodes for mesh in rings])
    conn = torch.cat((rings[0].elements,rings[1].elements+count))
    j = torch.arange(nt)
    inner,outer = count-nt+j,count+j
    fixed = torch.cat((j,2*count-nt+j))
    chord = 2*b*math.sin(math.pi/nt)
    load = torch.zeros_like(nodes[:,0])
    # Eliminate the massless sheet between the two series contact resistors.
    load[inner],load[outer] = chord*source*(1-fraction),chord*source*fraction
    polygon_heat = nt*chord*source+(pi_vol+po_vol)*np.sinc(2/nt)
    model = thermal.ThermalModel(nodes,conn,torch.from_numpy(np.repeat([ki,ko],n*nt)),1.,1.,
                       fixed,torch.zeros_like(load[fixed]),nodal_heat=load,element_type="q4",
                       heat_source=torch.from_numpy(np.repeat(bulk,n*nt)),
                       contact_pairs=torch.stack((inner,outer),dim=1),contact_conductance=chord/r)
    u = thermal.solve_steady_thermal(model,sparse=True)
    qi = (u[inner]-u[outer])/r-(1-fraction)*source
    surface = u[inner]-fraction*r*qi
    matrices = thermal.assemble_thermal(model,sparse=True)
    reactions = matrices.conductivity @ u-matrices.heat
    rates = np.array([float(reactions[j].sum()),-float(reactions[fixed[nt:]].sum())])
    rows = [dict(solver="FEM",sheet_error_K=float(abs(surface-sheet).max()),
                 rate_error_W_m=float(abs(rates-exact_rates).max()),
                 balance_W_m=float(abs(rates[1]-rates[0]-polygon_heat)))]
    for form in ("polygonal","circular"):
        result = layered_annulus.solve_annulus_contact((a,b,c),(ki,ko),(0.,0.),r,nt,formulation=form,
                                          interface_source=source,inner_resistance_fraction=fraction,volumetric_source=bulk)
        rates = np.array(result.heat_rates)[[0,-1]]
        generated = circ*source+pi_vol+po_vol if form=="circular" else polygon_heat
        rows.append(dict(solver="BEM_"+form,sheet_error_K=float(abs(result.sheet_temperature-sheet).max()),
                         rate_error_W_m=float(abs(rates-exact_rates).max()),
                         balance_W_m=float(abs(rates[1]-rates[0]-generated))))
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk",nargs=2,type=float,metavar=("INNER","OUTER"),default=(0.,0.),help="layer heat sources in W/m³")
    args = parser.parse_args()
    rows = []
    for fraction in (0.,.35,1.):
        previous = {}
        for n in (4,8,16,32):
            for row in measure(n,fraction,args.bulk):
                name = row["solver"]
                errors = np.array([row["sheet_error_K"],row["rate_error_W_m"]])
                order = np.log2(previous[name]/errors) if name in previous and name!="BEM_circular" else None
                passed = (max(errors)<1e-10 if name=="BEM_circular" else order is None or min(order)>1.7)
                balance_limit = .6/n**2 if name=="BEM_polygonal" else 1e-10
                passed = passed and row["balance_W_m"]<balance_limit
                row.update(volumetric_sources_W_m3=args.bulk,radial_cells_per_layer=n,panels_per_ring=4*n,inner_resistance_fraction=fraction,
                           orders=order.tolist() if order is not None else None,passed=bool(passed))
                rows.append(row)
                previous[name] = errors
    sources = [provenance(f) for f in (__file__,thermal.__file__,thermal_analytic.__file__,layered_annulus.__file__)]
    print(json.dumps(dict(runs=rows,sources=sources),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
