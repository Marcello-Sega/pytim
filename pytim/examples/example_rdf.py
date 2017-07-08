# vim: set expandtab:
# vim: set tabstop=4:
import MDAnalysis as mda
import numpy as np
import pytim
from pytim.datafiles import *
from pytim import observables

u = mda.Universe(WATER_GRO, WATER_XTC)
L = np.min(u.dimensions[:3])
oxygens = u.select_atoms("name OW")
radii = pytim_data.vdwradii(G43A1_TOP)

rdf = observables.RDF2D(u, max_radius='full', nbins=120)

interface = pytim.ITIM(u, alpha=2., itim_group=oxygens,
                       max_layers=4, radii_dict=radii, cluster_cut=3.5)

for ts in u.trajectory[::5]:
    print ("frame " + str(ts.frame) + " / " + str(len(u.trajectory)))
    layer = interface.layers[0, 0]
    rdf.sample(layer, layer)

rdf.rdf[0] = 0.0
np.savetxt('RDF.dat', np.column_stack((rdf.bins, rdf.rdf)))
