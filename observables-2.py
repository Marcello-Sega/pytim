import numpy as np
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *
from   pytim.observables import Profile
from matplotlib import pyplot as plt

u = mda.Universe(WATER_GRO,WATER_XTC)
g=u.select_atoms('name OW')
inter = pytim.ITIM(u,group=g,max_layers=4,centered=True)

Layers=[]
AIL = inter.atoms.in_layers

Layers.append(Profile(u.atoms))
for n in np.arange(4):
    Layers.append(Profile(AIL[::,n]))

for ts in u.trajectory[::]:
    for L in Layers:
        L.sample()

density=[]
for L in Layers:
    low,up,avg = L.get_values(binwidth=0.5)
    density.append(avg)


for dens in density:
    plt.plot(low,dens)

plt.gca().set_xlim([80,120])
plt.show()