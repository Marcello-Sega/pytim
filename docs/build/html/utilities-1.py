import numpy as np
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *
from matplotlib import pyplot as plt

u = mda.Universe(WATER_GRO,WATER_XTC)
inter = pytim.ITIM(u)

size=[]
time=[]
for ts in u.trajectory[:]:
    time.append(ts.time)
    size.append(len(inter.layers[0,0]))

corr =  pytim.utilities.correlate(size-np.mean(size))
plt.plot(time,corr/corr[0])
plt.plot(time,[0]*len(time))
plt.gca().set_xlabel("time/ps")

plt.show()