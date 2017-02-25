import MDAnalysis as mda
import numpy as np
import pytim 
import matplotlib.pyplot as plt
from   pytim.datafiles import *
u = mda.Universe(WATER_GRO,WATER_XTC)
L = np.min(u.dimensions[:3])
oxygens = u.select_atoms("name OW") 
radii=pytim_data.vdwradii(G43A1_TOP)
interface = pytim.ITIM(u,alpha=2.,itim_group=oxygens,max_layers=4,multiproc=True,radii_dict=radii,cluster_cut=3.5)
for ts in u.trajectory[::5] :
    interface.assign_layers()
    layer=interface.layers('upper',1)   
    if ts.frame==0 :
        rdf=pytim.observables.InterRDF2D(layer,layer,range=(0.,L/2.),nbins=120)
    rdf.sample(ts)
rdf.normalize()
rdf.rdf[0]=0.0
plt.plot(rdf.bins, rdf.rdf)
plt.show()