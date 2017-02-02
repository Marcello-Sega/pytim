import MDAnalysis as mda
import pytim 
from pytim.datafiles import *
u     = mda.Universe(WATER_GRO)
itim  = pytim.ITIM(u)
group =  u.select_atoms("name OW") 
itim.assign_layers()
layer = itim.layers[0][0]  # first layer
print layer
print group
