import MDAnalysis as mda
import pytim
import pytim.usti
from pytim.datafiles import WATER_GRO
import nglview
import numpy as np

# load the configuration in MDAnalysis
#u = mda.Universe(WATER_GRO)
u=mda.Universe("pytim/data/micelle.pdb") #Problem with weights!!! Weights are in Angstroemes but coordinates are in nm !!!!
g=u.select_atoms("name  OW1")

# compute the interface using ITIM. Identify 4 layers.
periodicity=np.array([1,1,1])
inter = pytim.usti.USTI(u,group=g,alpha=5,max_layers=4,max_interfaces=4,periodicity=periodicity)
inter.writepdb('waterOut.pdb')
print(len(inter.layers))


clusters=inter.clusters

print("cluster size","number of interfaces","dimension of cluster","cluster density")
for c in clusters:
    print(len(c.tetrahedrons),len(c.interfaces),c.clusterDimension,c.clusterDensity)

layers = inter.layers

