# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import numpy as np
from pytim import *
from pytim.datafiles import *

u = mda.Universe(WATER_GRO, WATER_XTC)
oxygens = u.select_atoms('name OW')
radii = pytim_data.vdwradii(G43A1_TOP)

obs = observables.Number(u)
profile = observables.Profile(observable=obs)

for ts in u.trajectory[:]:
  #  utilities.center(u, oxygens)
    profile.sample(oxygens)

lower, upper,  avg = profile.get_values(binwidth=1.0)
bins = (lower + upper) / 2.
np.savetxt('profile.dat', list(zip(bins, avg)))
