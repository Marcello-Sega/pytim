# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim
from pytim.datafiles import *

u = mda.Universe(WATER_GRO)
#oxygens = u.select_atoms("name OW")
#g = oxygens
#radii = pytim_data.vdwradii(G43A1_TOP)
interface = pytim.ITIM(u, alpha=2., max_layers=4, molecular=True)
layer = interface.layers[0, 0]  # first layer, upper side
print(repr(interface.layers[0, 0]))

interface.writepdb('layers.pdb', centered=False)

# for pytest
def test(): assert repr(interface.layers[0,0]) == '<AtomGroup with 651 atoms>'
