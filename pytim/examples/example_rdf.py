# vim: set expandtab:
# vim: set tabstop=4:
import MDAnalysis as mda
import numpy as np
import pytim
from pytim.datafiles import *
from pytim import observables


##########################
sampling_frequency = 50  # change this to 1 to sample each frame
##########################

u = mda.Universe(WATER_GRO, WATER_XTC)
L = np.min(u.dimensions[:3])
oxygens = u.select_atoms("name OW")
radii = pytim_data.vdwradii(G43A1_TOP)

rdf = observables.RDF2D(u, max_radius='full', nbins=120)

interface = pytim.ITIM(u, alpha=2., group=oxygens,
                       max_layers=4, radii_dict=radii, cluster_cut=3.5)

for ts in u.trajectory[::sampling_frequency]:
    print("frame " + str(ts.frame) + " / " + str(len(u.trajectory)))
    layer = interface.layers[0, 0]
    rdf.sample(layer, layer)

rdf.rdf[0] = 0.0
np.savetxt('RDF.dat', np.column_stack((rdf.bins, rdf.rdf)))
print('RDF saved to RDF.dat')
if sampling_frequency > 1:
    print('set sampling_frequency  = 1 in order to sample each frame in the trajectory')
