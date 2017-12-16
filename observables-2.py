from matplotlib import pyplot as plt

import numpy as np
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *
from   pytim.observables import Profile

u = mda.Universe(WATER_GRO,WATER_XTC)
g=u.select_atoms('name OW')
# here we calculate the profiles of oxygens only (note molecular=False)
inter = pytim.ITIM(u,group=g,max_layers=4,centered=True, molecular=False)

Layers=[]
# by default Profile() uses the number of atoms as an observable
for n in np.arange(0,5):
    Layers.append(Profile())

for ts in u.trajectory[::50]:
    for n in range(len(Layers)):
        if n>0:
            group = u.atoms[u.atoms.layers == n ]
        else:
            group = g
        Layers[n].sample(group)

for L in Layers:
    low,up,avg = L.get_values(binwidth=0.5)
    plt.plot(low,avg)

plt.gca().set_xlim([80,120])
plt.show()