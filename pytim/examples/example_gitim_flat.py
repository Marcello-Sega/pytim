# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim
from pytim.datafiles import *

u = mda.Universe(WATER_GRO)

g = u.select_atoms('name OW')

interface = pytim.GITIM(u, group=g, molecular=True, symmetry='planar',
                        alpha=2.5, cluster_cut=3.5, cluster_threshold_density='auto')

layers = interface.layers[:]
print(repr(layers[0]))

interface.writepdb('gitim_flat.pdb', centered='middle')
