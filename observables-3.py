import MDAnalysis as mda
import numpy as np
import pytim
import matplotlib.pyplot as plt
from   pytim.datafiles   import *

u       = mda.Universe(WATER_GRO,WATER_XTC)
oxygens = u.select_atoms("name OW")

obs     = pytim.observables.Number()

interface = pytim.ITIM(u, alpha=2.0, max_layers=1,cluster_cut=3.5,centered=True,molecular=False)
profile = pytim.observables.Profile(group=oxygens,observable=obs,interface=interface)

for ts in u.trajectory[::]:
    profile.sample()

low, up, avg = profile.get_values(binwidth=0.2)
z = (low+up)/2.
plt.plot(z, avg)
axes = plt.gca()
axes.set_xlim([-20,20])
axes.set_ylim([0,0.1])
plt.show()