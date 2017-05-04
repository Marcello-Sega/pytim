# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim
from   pytim.datafiles import *

u         = mda.Universe(MICELLE_PDB)
g         = u.select_atoms('resname DPC')

radii     = pytim_data.vdwradii(G43A1_TOP)

interface = pytim.WillardChandler(u,itim_group=g,alpha=3.0,
                                   density_basename="dens",
                                   particles_basename="atoms",
                                   surface_basename="surf")

