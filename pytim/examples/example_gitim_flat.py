# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *

u         = mda.Universe(WATER_GRO)

radii     = pytim_data.vdwradii(G43A1_TOP)

interface = pytim.GITIM(u,molecular=True,symmetry='planar',alpha=2.5, cluster_cut=3.5, cluster_threshold_density='auto')

layers     = interface.layers[:]
print layers

interface.writepdb('gitim_flat.pdb',centered='middle')


