# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import print_function


def PatchTrajectory(trajectory, interface):
    """ Patch the MDAnalysis trajectory class

        this patch makes the layer assignement being automatically
        called whenever a new frame is loaded.
    """
    try:
        trajectory.interface
    except AttributeError:
        trajectory.interface = interface
        trajectory.original_read_next_timestep = trajectory._read_next_timestep

        trajectory.original_read_frame_with_aux = trajectory._read_frame_with_aux

        class PatchedTrajectory(trajectory.__class__):
            def _read_next_timestep(self, ts=None):
                tmp = self.original_read_next_timestep(ts=ts)
                if self.interface.autoassign is True:
                    self.interface._assign_layers()
                return tmp

            def _read_frame_with_aux(self, frame):
                if frame != self.frame:
                    tmp = self.original_read_frame_with_aux(frame)
                    if self.interface.autoassign is True:
                        self.interface._assign_layers()
                    return tmp
                return self.ts

        oldname = trajectory.__class__.__name__
        oldmodule = trajectory.__class__.__module__

        PatchedTrajectory.__name__ = oldname
        PatchedTrajectory.__module__ = oldmodule
        trajectory.__class__ = PatchedTrajectory


def PatchOpenMM(simulation, interface):
    #       return _openmm.CompoundIntegrator_step(self, steps)
    """ Patch the Openmm Simulation class

    """
    from simtk.unit import angstrom as openmm_AA
    try:
        simulation.interface
    except AttributeError:
        simulation.interface = interface
        simulation.original_step = simulation.step

        class PatchedOpenMMSimulation(simulation.__class__):
            def step(self, steps):
                tmp = self.original_step(steps)
                pos = self.context.getState(getPositions=True).getPositions(
                    asNumpy=True).value_in_unit(openmm_AA)
                self.interface.universe.atoms.positions = pos
                if self.interface.autoassign is True:
                    self.interface._assign_layers()
                return tmp

        oldname = simulation.__class__.__name__
        oldmodule = simulation.__class__.__module__

        PatchedOpenMMSimulation.__name__ = oldname
        PatchedOpenMMSimulation.__module__ = oldmodule
        simulation.__class__ = PatchedOpenMMSimulation


def PatchMDTRAJ(trajectory, universe):
    """ Patch the mdtraj Trajectory class

        automates the data exchange between MDAnalysis and mdtraj classes
    """
    try:
        trajectory.universe
    except AttributeError:
        trajectory.universe = universe

        class PatchedMdtrajTrajectory(trajectory.__class__):
            def __getitem__(self, key):
                slice_ = self.slice(key)
                PatchMDTRAJ(slice_, universe)
                if isinstance(key, int):
                    # mdtraj uses nm as distance unit, we need to convert to
                    # Angstrom for MDAnalysis
                    slice_.universe.atoms.positions = slice_.xyz[0] * 10.0
                    dimensions = slice_.universe.dimensions[:]
                    dimensions[0:3] = slice_.unitcell_lengths[0:3] * 10.0
                    slice_.universe.dimensions = dimensions
                    if slice_.universe.trajectory.interface.autoassign is True:
                        slice_.universe.trajectory.interface._assign_layers()
                return slice_

        oldname = trajectory.__class__.__name__
        oldmodule = trajectory.__class__.__module__

        PatchedMdtrajTrajectory.__name__ = oldname
        PatchedMdtrajTrajectory.__module__ = oldmodule
        trajectory.__class__ = PatchedMdtrajTrajectory
