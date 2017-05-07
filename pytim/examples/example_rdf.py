# vim: set expandtab:
# vim: set tabstop=4:
import MDAnalysis as mda
import numpy as np
import pytim
from pytim.datafiles import *
from pytim.observables import *

u = mda.Universe(WATER_GRO, WATER_XTC)
L = np.min(u.dimensions[:3])
oxygens = u.select_atoms("name OW")
radii = pytim_data.vdwradii(G43A1_TOP)

interface = pytim.ITIM(u, alpha=2., itim_group=oxygens,
                       max_layers=4, radii_dict=radii, cluster_cut=3.5)

for ts in u.trajectory[::5]:
    print ("frame " + str(ts.frame) + " / " + str(len(u.trajectory)))
    layer = interface.layers('upper', 1)
    if ts.frame == 0:
        rdf = InterRDF2D(layer, layer, range=(0., L / 2.), nbins=120)
    rdf.sample(ts)

rdf.normalize()
rdf.rdf[0] = 0.0
np.savetxt('RDF.dat', np.column_stack((rdf.bins, rdf.rdf)))
