import MDAnalysis as mda
from MDAnalysis.core.AtomGroup   import *

import pytim

u = mda.Universe("../data/water.gro")
itim  = pytim.ITIM(u,alpha=2.0)

group =  u.select_atoms("name OW")

itim.assign_layers()

layer = itim.layers[0][0]

print group
print layer


