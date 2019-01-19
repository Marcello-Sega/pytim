# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
"""
    This example shows how to use pytim classes online during
    a simulation performed with openmm (http://mdtraj.org/)

    (see also the mdtraj interoperability)
"""
# openmm imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
# pytim
import pytim
from pytim.datafiles import WATER_PDB

# usual openmm setup, we load one of pytim's example files
pdb = PDBFile(WATER_PDB)
forcefield = ForceField('amber99sb.xml', 'spce.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
                                 nonbondedCutoff=1 * nanometer)
integrator = LangevinIntegrator(
    300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# just pass the openmm Simulation object to pytim
inter = pytim.ITIM(simulation)
print(repr(inter.atoms))

# the new interfacial atoms will be computed at the end
# of the integration cycle
simulation.step(10)
print(repr(inter.atoms))
