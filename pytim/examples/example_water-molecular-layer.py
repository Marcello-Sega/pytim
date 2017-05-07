# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import numpy as np
import pytim
from pytim.datafiles import *

u = mda.Universe(WATER_GRO)
radii = pytim_data.vdwradii(G43A1_TOP)

interface = pytim.ITIM(u, max_layers=4, molecular=True, cluster_cut=3.5)
interface.writepdb('layers.pdb')
