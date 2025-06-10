# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

"""
    This example shows how to use pytim classes on trajectories
    loaded with MDTraj (http://mdtraj.org/)
    (see also the openmm interoperability)
"""
import mdtraj
import pytim
from pytim.datafiles import WATER_GRO, WATER_XTC

t = mdtraj.load_xtc(WATER_XTC, top=WATER_GRO)
inter = pytim.ITIM(t)
for step in t[[1,-1]]:
    print("surface atoms: "+repr(inter.atoms.indices))

# for pytest
def test(): assert len(inter.atoms.indices)==1506

