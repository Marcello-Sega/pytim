import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *
from   pytim.observables import Profile

u = mda.Universe(WATER_GRO,WATER_XTC)
g = u.select_atoms("name OW")

inter = pytim.ITIM(u, group=g,max_layers=1,cluster_cut=3.5,centered=True, molecular=False)
profile = Profile(interface=inter)

for ts in u.trajectory[::50]:
    profile.sample(g)

low, up, avg = profile.get_values(binwidth=0.2)

z = (low+up)/2.
plt.plot(z, avg)
axes = plt.gca()
axes.set_xlim([-15,5])
axes.set_ylim([0,0.05])
plt.show()