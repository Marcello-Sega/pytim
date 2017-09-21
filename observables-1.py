import MDAnalysis as mda
import numpy as np
import pytim
import matplotlib.pyplot as plt
from   pytim.datafiles import *
u = mda.Universe(WATER_GRO,WATER_XTC)
oxygens = u.select_atoms("name OW")
interface = pytim.ITIM(u,alpha=2.,group=oxygens,                               cluster_cut=3.5, molecular=False)
rdf=pytim.observables.RDF2D(u,nbins=250)
for ts in u.trajectory[::] :
    layer=interface.layers[0,0]
    rdf.sample(layer,layer)
rdf.count[0]=0

plt.plot(rdf.bins, rdf.rdf)

plt.gca().set_xlim([0,7])

plt.show()