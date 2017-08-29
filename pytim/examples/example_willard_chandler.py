# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
import MDAnalysis as mda
import pytim
from pytim.datafiles import *

u = mda.Universe(MICELLE_PDB)
g = u.select_atoms('resname DPC')

interface = pytim.WillardChandler(u, group=g, alpha=3.0)

interface.writecube('density.cube') # volumetric density in cube format
interface.writevtk.density('dens.vtk') # volumetric density in vtk format  

interface.writecube('density_and_particles.cube',group=g) # cube format including volumetric density and particles
interface.writevtk.particles('particles.vtk') # particles in vtk format  

interface.writeobj('surface.obj') # isodensity surface in wavefront obj format
interface.writevtk.surface('surface.vtk') # isodensity surface in vtk format

R, _, _, _ = pytim.utilities.fit_sphere(interface.triangulated_surface[0])
print "Radius={:.3f}".format(R)


