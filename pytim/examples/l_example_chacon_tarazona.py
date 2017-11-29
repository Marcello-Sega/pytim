import MDAnalysis as mda
import numpy as np
import pytim
from pytim.datafiles import *
from pytim import observables

u = mda.Universe(WATER_GRO)
g = u.select_atoms('name OW')
inter = pytim.ChaconTarazona(
    u, alpha=2., tau=1.5, group=g, info=True, molecular=False)
inter.writepdb('CT.pdb', centered=True)
print repr(inter.layers)
exit()
number = observables.Number()
profile = observables.Profile(group=g, observable=number, interface=inter)
profile.sample()

print inter.layers[:]
low, up, avg = profile.get_values(binwidth=0.1)
bins = (low + up) / 2.
np.savetxt('intrdist.dat', list(zip(bins, avg)), fmt=['%.5f', '%e'])

try:
    import matplotlib.pyplot as plt
    plt.plot(bins, avg)
    maxval = np.max(avg[:len(bins) / 2 - 1])
    plt.ylim((0, maxval * 1.5))
    plt.xlim((-20, 10))
    plt.savefig("intrinsic.pdf")
    print("Intrinsic profile saved in intrinsic.pdf")
    plt.show()
except:
    raise Warning("matplotlib failed")
