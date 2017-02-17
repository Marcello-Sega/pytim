# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import numpy as np
import pytim  
from pytim.datafiles import *

u       = mda.Universe(WATER_GRO,WATER_GRO)
oxygens = u.select_atoms("name OW") 
radii=pytim_data.vdwradii(G43A1_TOP)

interface =pytim.ITIM(u,alpha=2.,itim_group=oxygens,max_layers=4)#,multiproc=True,radii_dict=radii,cluster_groups=oxygens,cluster_cut=3.5)
pytim.lap()
interface.assign_layers()
pytim.lap()
layer = interface.layers('upper',1)  # first layer, upper side
print ("Interface computed. Upper layer:\n %s out of %s" % (layer,oxygens))

interface.writepdb('layers.pdb')

