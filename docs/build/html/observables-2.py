import MDAnalysis as mda
import numpy as np
import pytim  
import matplotlib.pyplot as plt
from   pytim.datafiles   import *

u       = mda.Universe(WATER_GRO,WATER_XTC)
oxygens = u.select_atoms("name OW") 
radii=pytim_data.vdwradii(G43A1_TOP)

obs     = pytim.observables.Number()
profile = pytim.observables.Profile(group=oxygens,observable=obs)

interface = pytim.ITIM(u, alpha=2.0, max_layers=1,cluster_cut=3.5) 

for ts in u.trajectory[:]:
    interface.center(oxygens)
    profile.sample()

bins, avg = profile.profile(binwidth=1.0)
plt.plot(bins, avg)
plt.show()