# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim
from pytim.datafiles import *

u = mda.Universe(MICELLE_PDB)
g = u.select_atoms('resname DPC')

radii = pytim_data.vdwradii(G43A1_TOP)

interface = pytim.GITIM(u, itim_group=g, molecular=False,
                        symmetry='spherical', alpha=2.5,)

layer = interface.layers[0]

interface.writepdb('gitim.pdb', centered=False)

print("selected one layer of " + str(len(layer)) +
      " atoms out of a group of " + str(len(g)))
