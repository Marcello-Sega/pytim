import MDAnalysis as mda
import pytim 
from pytim.datafiles import *

u       = mda.Universe(WATER_GRO)
oxygens = u.select_atoms("name OW") 

interface = pytim.ITIM(u,alpha=2.,itim_group=oxygens,mesh=0.1)
interface.assign_layers()

layer = interface.layers('upper',1)  # first layer
print ("Interface computed. Upper layer:\n %s out of %s" % (layer,oxygens))

interface.writepdb('layers.pdb')
