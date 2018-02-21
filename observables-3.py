import pytim
import MDAnalysis as mda
import numpy as np
from pytim.datafiles import WATERSMALL_GRO
from pytim.utilities import lap
WATERSMALL_TRR=pytim.datafiles.pytim_data.fetch('WATERSMALL_LONG_TRR')

u = mda.Universe(WATERSMALL_GRO,WATERSMALL_TRR)
g = u.select_atoms('name OW')

velocity = pytim.observables.Velocity()
corr = pytim.observables.Correlator(observable=velocity)
for t in u.trajectory[1:]:
    corr.sample(g)

vacf = corr.correlation()

from matplotlib import pyplot as plt
plt.plot(vacf[:1000])
plt.plot([0]*1000)
plt.show()