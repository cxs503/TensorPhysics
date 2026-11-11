import sys, os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
sys.path.insert(0,'src')
from tensorphysics.grid_to_grid_remap.overlap import build_overlap_table
from tensorphysics.grid_to_grid_remap.remap import remap_cell_average, remap_p1_nodal, l2_error
from tensorphysics.grid_to_grid_remap.acceptance import make_perturbed_tri_mesh
f=lambda xy: np.sin(np.pi*xy[:,0])*np.sin(np.pi*xy[:,1])
src=make_perturbed_tri_mesh(32,jitter=0.30,seed=7); tgt=make_perturbed_tri_mesh(48,jitter=0.20,seed=107)
tab=build_overlap_table(src,tgt); phi=f(src.centroids())
ra=remap_cell_average(tab,phi); rb=remap_p1_nodal(tab,phi,mode="bounded")
ref=f(tgt.centroids())
os.makedirs("artifacts/m15004",exist_ok=True)
fig,ax=plt.subplots(2,2,figsize=(10,8))
for a,(m,val,t) in zip(ax.ravel(),[(src,phi,"source 32x32 perturbed"),(tgt,ra.phi,"average remap -> 48x48"),(tgt,rb.phi,"bounded remap -> 48x48"),(tgt,rb.phi-ref,"bounded - analytic (point values)")]):
    tp=mtri.Triangulation(m.nodes[:,0],m.nodes[:,1],m.tri)
    cs=a.tripcolor(tp,facecolors=val,cmap="viridis" if "analytic" not in t else "RdBu_r")
    a.set_title(t,fontsize=9); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(cs,ax=a,fraction=0.046)
fig.suptitle("m14679 GAP-CROSS-MAP: field transfer audit (perturbed 32x32 -> 48x48)",fontsize=10)
fig.tight_layout()
out="artifacts/m15004/m14679_field_transfer_audit.png"; fig.savefig(out,dpi=150); print("wrote",out)
d=json.load(open("runs/m15004_audit_m14679/checks.json"))
d["G_field"]={"avg_l2_rel":l2_error(tab,ra.phi,ref)["l2_rel"],"bnd_l2_rel":l2_error(tab,rb.phi,ref)["l2_rel"],
              "max_abs_bnd_err":float(np.max(np.abs(rb.phi-ref))),"figure":out}
json.dump(d,open("runs/m15004_audit_m14679/checks.json","w"),indent=2)
print(d["G_field"])
