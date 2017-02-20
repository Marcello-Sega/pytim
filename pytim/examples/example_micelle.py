# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim  
from   pytim.datafiles import *

u       = mda.Universe(MICELLE_PDB)
g       = u.select_atoms('resname DPC')
print g
radii=pytim_data.vdwradii(G43A1_TOP)

interface =pytim.GITIM(u,itim_group=g,molecular=False,symmetry='spherical',alpha=4.0)
interface.assign_layers()

interface.writepdb('test.pdb')

