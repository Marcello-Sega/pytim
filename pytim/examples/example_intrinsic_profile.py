# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import numpy as np
import pytim
from   pytim import observables
from   pytim.datafiles import *

u         = mda.Universe(WATER_GRO,WATER_XTC)
oxygens   = u.select_atoms("name OW")
radii     = pytim_data.vdwradii(G43A1_TOP)

number    = observables.Number()
interface = pytim.ITIM(u,alpha=2.)

profile   = observables.Profile(group=oxygens,observable=number, interface=interface)

for t in u.trajectory[::]:
    print t.frame
    profile.sample()

bins, avg = profile.profile(binwidth=0.1)
np.savetxt('intrdist.dat',list(zip(bins,avg)),fmt=['%.5f','%e'])

# the maximum, excluding the delta contribution

values = np.loadtxt('intrdist.dat')
import matplotlib.pyplot as plt

plt.plot(values[:,0],values[:,1])

maxval = np.max(avg[:len(bins)/2-1])
plt.ylim((0,maxval*1.5))
plt.xlim((-20,10))

try:
    # nice latex labels for publication-ready figures, in case
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$\xi/\AA$')
    plt.ylabel(r'$\rho  \AA^3$')
except:
    pass

plt.savefig("intrinsic.pdf")
print("Intrinsic profile saved in intrinsic.pdf")
plt.show()



